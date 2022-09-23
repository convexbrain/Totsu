//! [`F32CUDASlice`] module.

use std::prelude::v1::*;
use std::rc::Rc;
use std::cell::RefCell;
use std::thread_local;
use std::collections::HashMap;
use std::pin::Pin;
use rustacuda::prelude::*;
use rustacuda::memory::{DeviceBuffer, DeviceSlice};
use totsu_core::solver::{SliceRef, SliceMut, SliceLike};
use super::cuda_mgr;

#[derive(Eq, PartialEq, Copy, Clone)]
enum CUDASliceMut
{
    Sync,
    Host,
    Dev,
}

#[derive(Eq, PartialEq, Copy, Clone)]
enum HostBuf
{
    Ref(*const f32),
    Mut(*mut f32),
}

/// `f32`-specific slice with CUDA device buffer, [`SliceLike`] implementation for [`crate::F32CUDA`]`::Sl`.
/// 
/// The CUDA buffer and the host buffer are internally held.
/// Their contents are synchronized whenever necessary.
pub struct F32CUDASlice
{
    idx: usize,
    parent_idx: Option<usize>,
    dev_buf: Rc<RefCell<DeviceBuffer<f32>>>,
    host_buf: HostBuf,
    sta: usize,
    end: usize,
    mutator: RefCell<CUDASliceMut>,
}

struct SliceManager
{
    cnt: usize,
    map: HashMap<usize, Pin<Box<F32CUDASlice>>>,
}

impl SliceManager
{
    fn new() -> SliceManager
    {
        SliceManager {
            cnt: 0,
            map: HashMap::new()
        }
    }

    fn new_slice<'a, F>(&mut self, func: F) -> &'a mut F32CUDASlice
    where F: FnOnce(usize) -> F32CUDASlice
    {
        let idx = self.cnt;
        self.cnt = idx + 1;

        let cs = func(idx);

        let r = self.map.insert(idx, Box::pin(cs));
        assert!(r.is_none());
        let cs = self.map.get_mut(&idx).unwrap();

        unsafe {
            std::mem::transmute::<&mut F32CUDASlice, &'a mut F32CUDASlice>(cs)
        }
    }
}

impl Drop for SliceManager
{
    fn drop(&mut self) {
        if self.map.len() > 0 {
            log::warn!("Memory leak: {} slices", self.map.len())
        }
    }
}

//

thread_local!(static SLICE_MANAGER: RefCell<SliceManager> = RefCell::new(SliceManager::new()));

fn new_slice_from_ref(s: &[f32]) -> &mut F32CUDASlice
{
    SLICE_MANAGER.with(|mgr| {
        let mut mgr = mgr.borrow_mut();

        mgr.new_slice(|idx| {
            F32CUDASlice {
                idx,
                parent_idx: None,
                dev_buf: Rc::new(RefCell::new(cuda_mgr::buf_from_slice(s))),
                host_buf: HostBuf::Ref(s.as_ptr()),
                sta: 0,
                end: s.len(),
                mutator: RefCell::new(CUDASliceMut::Sync)
            }
        })
    })
}

fn new_slice_from_mut(s: &mut[f32]) -> &mut F32CUDASlice
{
    SLICE_MANAGER.with(|mgr| {
        let mut mgr = mgr.borrow_mut();

        mgr.new_slice(|idx| {
            F32CUDASlice {
                idx,
                parent_idx: None,
                dev_buf: Rc::new(RefCell::new(cuda_mgr::buf_from_slice(s))),
                host_buf: HostBuf::Mut(s.as_mut_ptr()),
                sta: 0,
                end: s.len(),
                mutator: RefCell::new(CUDASliceMut::Sync)
            }
        })
    })
}

fn split_slice(cs: &F32CUDASlice, sta: usize, end: usize) -> &mut F32CUDASlice
{
    assert!(sta <= end);
    assert!(cs.sta + sta <= cs.end);
    assert!(cs.sta + end <= cs.end);

    SLICE_MANAGER.with(|mgr| {
        let mut mgr = mgr.borrow_mut();

        mgr.new_slice(|idx| {
            F32CUDASlice {
                idx,
                parent_idx: Some(cs.idx),
                dev_buf: cs.dev_buf.clone(),
                host_buf: cs.host_buf,
                sta: cs.sta + sta,
                end: cs.sta + end,
                mutator: RefCell::new(*cs.mutator.borrow())
            }
        })
    })
}

fn remove_slice(idx: usize)
{
    SLICE_MANAGER.with(|mgr| {
        let mut mgr = mgr.borrow_mut();

        // sync mutators of split tree
        //
        // parent    child(cs)
        // Sync   -> Sync       do nothing
        //        -> Host       change parent to Host
        //        -> Dev        change parent to Dev
        // Host   -> Sync       do nothing
        //        -> Host       do nothing
        //        -> Dev        copy from Dev to Host
        // Dev    -> Sync       do nothing
        //        -> Host       copy from Host to Dev
        //        -> Dev        do nothing


        let cs = &mgr.map[&idx];
        let cs_mut = *cs.mutator.borrow();
        let cs_par_idx = cs.parent_idx;
        let mut cs_par_mut = None;

        if let Some(par_idx) = cs_par_idx {
            let par = mgr.map.get_mut(&par_idx).unwrap();
            let mut par_mut = par.mutator.borrow_mut();

            cs_par_mut = Some(*par_mut);

            if *par_mut == CUDASliceMut::Sync {
                *par_mut = cs_mut;
            }
        }

        let cs = mgr.map.get_mut(&idx).unwrap();
        if let Some(par_mut) = cs_par_mut {
            match par_mut {
                CUDASliceMut::Sync => {},
                CUDASliceMut::Host => {
                    if cs_mut == CUDASliceMut::Dev {
                        cs.sync_from_dev();
                    }
                },
                CUDASliceMut::Dev => {
                    if cs_mut == CUDASliceMut::Host {
                        cs.sync_from_host();
                    }
                },
            }
        }
        else {
            if cs_mut == CUDASliceMut::Dev {
                cs.sync_from_dev();
            }
        }

        mgr.map.remove(&idx).unwrap();
    });
}

//

impl SliceLike for F32CUDASlice
{
    type F = f32;

    fn new_ref(s: &[f32]) -> SliceRef<'_, F32CUDASlice>
    {
        let cs = new_slice_from_ref(s);
        
        //std::println!("{} new", cs.idx);
        unsafe { SliceRef::new(cs) }
    }

    fn new_mut(s: &mut[f32]) -> SliceMut<'_, F32CUDASlice>
    {
        let cs = new_slice_from_mut(s);

        //std::println!("{} new_mut", cs.idx);
        unsafe { SliceMut::new(cs) }
    }
    
    fn split_ref(&self, mid: usize) -> (SliceRef<'_, F32CUDASlice>, SliceRef<'_, F32CUDASlice>)
    {
        let cs0 = split_slice(self, 0, mid);
        let cs1 = split_slice(self, mid, self.len());

        //std::println!("{} split from {}", cs0.idx, self.idx);
        //std::println!("{} split from {}", cs1.idx, self.idx);
        unsafe { (SliceRef::new(cs0), SliceRef::new(cs1)) }
    }

    fn split_mut(&mut self, mid: usize) -> (SliceMut<'_, F32CUDASlice>, SliceMut<'_, F32CUDASlice>)
    {
        let cs0 = split_slice(self, 0, mid);
        let cs1 = split_slice(self, mid, self.len());

        //std::println!("{} split_mut from {}", cs0.idx, self.idx);
        //std::println!("{} split_mut from {}", cs1.idx, self.idx);
        unsafe { (SliceMut::new(cs0), SliceMut::new(cs1)) }
    }

    fn drop(&self)
    {
        //std::println!("{} drop", self.idx);
        remove_slice(self.idx);
    }

    fn len(&self) -> usize
    {
        //std::println!("  {} len {}", self.idx, self.end - self.sta);
        self.end - self.sta
    }

    fn get_ref(&self) -> &[f32]
    {
        //std::println!("  {} get", self.idx);
        let mut mutator = self.mutator.borrow_mut();
        //std::println!("  {:?} mutator", *mutator);
        match *mutator {
            CUDASliceMut::Sync | CUDASliceMut::Host => {},
            CUDASliceMut::Dev => {
                self.sync_from_dev();
                *mutator = CUDASliceMut::Sync;
            },
        }

        let hb_ref = self.host_buf_ref();

        unsafe {
            std::mem::transmute::<&[f32], &[f32]>(hb_ref)
        }
    }

    fn get_mut(&mut self) -> &mut[f32]
    {
        //std::println!("  {} get_mut", self.idx);
        let mut mutator = self.mutator.borrow_mut();
        //std::println!("    {:?} mutator", *mutator);
        match *mutator {
            CUDASliceMut::Sync => {
                *mutator = CUDASliceMut::Host;
            },
            CUDASliceMut::Host => {},
            CUDASliceMut::Dev => {
                self.sync_from_dev();
                *mutator = CUDASliceMut::Host;
            },
        }

        let hb_mut = self.host_buf_mut();

        unsafe {
            std::mem::transmute::<&mut[f32], &mut[f32]>(hb_mut)
        }
    }
}

//

impl F32CUDASlice
{
    fn host_buf_ref(&self) -> &[f32]
    {
        let hb_ptr = match self.host_buf {
            HostBuf::Ref(p) => {p},
            HostBuf::Mut(p) => {p as *const f32},
        };

        unsafe {
            let hb = std::ptr::slice_from_raw_parts(hb_ptr.offset(self.sta as isize), self.end - self.sta);

            hb.as_ref().unwrap()
        }
    }

    fn host_buf_mut(&self) -> &mut[f32]
    {
        let hb_ptr = match self.host_buf {
            HostBuf::Ref(_) => {panic!()},
            HostBuf::Mut(p) => {p},
        };

        unsafe {
            let hb = std::ptr::slice_from_raw_parts_mut(hb_ptr.offset(self.sta as isize), self.end - self.sta);

            hb.as_mut().unwrap()
        }
    }

    fn sync_from_dev(&self)
    {
        let db = &self.dev_buf.as_ref().borrow()[self.sta..self.end];
        let hb_mut = self.host_buf_mut();
        db.copy_to(hb_mut).unwrap();
    }

    fn sync_from_host(&self)
    {
        let db = &mut self.dev_buf.as_ref().borrow_mut()[self.sta..self.end];
        let hb_ref = self.host_buf_ref();
        db.copy_from(hb_ref).unwrap();
    }

    /// Returns a reference of the content CUDA device slice.
    pub fn get_dev(&self) -> &DeviceSlice<f32>
    {
        let mut mutator = self.mutator.borrow_mut();
        match *mutator {
            CUDASliceMut::Sync | CUDASliceMut::Dev => {},
            CUDASliceMut::Host => {
                self.sync_from_host();
                *mutator = CUDASliceMut::Sync;
            },
        }

        let db_ref = &self.dev_buf.as_ref().borrow()[self.sta..self.end];

        unsafe {
            std::mem::transmute::<&DeviceSlice<f32>, &DeviceSlice<f32>>(db_ref)
        }
    }

    /// Mutable version of [`F32CUDASlice::get_dev`].
    pub fn get_dev_mut(&mut self) -> &mut DeviceSlice<f32>
    {
        let mut mutator = self.mutator.borrow_mut();
        match *mutator {
            CUDASliceMut::Sync => {
                *mutator = CUDASliceMut::Dev;
            },
            CUDASliceMut::Dev => {},
            CUDASliceMut::Host => {
                self.sync_from_host();
                *mutator = CUDASliceMut::Dev;
            },
        }

        let db_mut = &mut self.dev_buf.as_ref().borrow_mut()[self.sta..self.end];

        unsafe {
            std::mem::transmute::<&mut DeviceSlice<f32>, &mut DeviceSlice<f32>>(db_mut)
        }
    }
}
