QEMU='qemu-system-gnuarmeclipse'  # xPack/QEMU ARM/2.8.0-8/bin/

"$QEMU" -board NUCLEO-F103RB -nographic -image $1 | tee log_qemu.txt
