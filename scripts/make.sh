# ==============================================
# Build this project according to Cmake rules
# ==============================================
# Create build directories if need
mkdir -p build
mkdir -p bin
mkdir -p lib
# Clean up script
scripts/clean.sh
cd build/
# Invoke Cmake and set as release mode
cmake .. -DCMAKE_BUILD_TYPE=Release 
# Compile targets
make -j12
cd ..
