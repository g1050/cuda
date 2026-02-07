set -e

cmake -S . -B ./build
cmake --build ./build -j

pushd build
ctest -R test_ir -V
# ctest
popd

# build/test_tensor

# cp ./build/compile_commands.json ../