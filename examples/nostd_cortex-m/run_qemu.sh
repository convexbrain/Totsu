QEMU='/c/NoInstaller/xPack/QEMU ARM/2.8.0-8/bin/qemu-system-gnuarmeclipse'

"$QEMU" -board NUCLEO-F103RB -nographic -image $1
