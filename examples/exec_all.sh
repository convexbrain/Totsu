
(cd l1reg_lp         && pwd && RUST_LOG=debug cargo run --release) || exit
(cd svm_qp           && pwd && RUST_LOG=debug cargo run --release) || exit
(cd trajplan_qcqp    && pwd && RUST_LOG=debug cargo run --release) || exit
(cd toruscompl_socp  && pwd && RUST_LOG=debug cargo run --release) || exit
(cd partitioning_sdp && pwd && RUST_LOG=debug cargo run --release) || exit
(cd imgnr_udef       && pwd && RUST_LOG=debug cargo run --release) || exit

(cd nostd_cortex-m && pwd && cargo build --release && sh run_qemu.sh target/thumbv7m-none-eabi/release/nostd_cortex-m) || exit

echo
echo "=================="
echo "End of exec_all.sh"
echo "=================="
