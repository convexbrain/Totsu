
(cd l1reg_lp         && RUST_LOG=debug cargo run --release) || exit
(cd svm_qp           && RUST_LOG=debug cargo run --release) || exit
(cd trajplan_qcqp    && RUST_LOG=debug cargo run --release) || exit
(cd toruscompl_socp  && RUST_LOG=debug cargo run --release) || exit
(cd partitioning_sdp && RUST_LOG=debug cargo run --release) || exit

(cd nostd_cortex-m && cargo build --release && sh run_qemu.sh target/thumbv7m-none-eabi/release/nostd_cortex-m) || exit

echo "==="
echo "End"
echo "==="
