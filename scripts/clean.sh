# ================================
# Clean up any generated files 
# ================================
# Out-of-source build simply removes everything which is generated
#shopt -s extglob
#rm -rf !(*.sh)   
rm -rf build/*
rm -rf bin/*
#rm -rf lib/*

# In-source build needs to be careful about what to remove
# Add -f flag to suppress warning of 'no such file or directory'
#rm -f *.exe
#rm -f *.so
#rm -rf *_automoc.*
#rm -rf CMakeCache.txt
#rm -rf CMakeFiles
#rm -rf cmake_install.cmake
#rm -rf Makefile
