#
# CMakeLists.txt
# RVO2 Library
#
# Copyright 2008 University of North Carolina at Chapel Hill
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Please send all bug reports to <geom@cs.unc.edu>.
#
# The authors may be contacted via:
#
# Jur van den Berg, Stephen J. Guy, Jamie Snape, Ming C. Lin, Dinesh Manocha
# Dept. of Computer Science
# 201 S. Columbia St.
# Frederick P. Brooks, Jr. Computer Science Bldg.
# Chapel Hill, N.C. 27599-3175
# United States of America
#
# <http://gamma.cs.unc.edu/RVO2/>
#

cmake_minimum_required(VERSION 3.16)
project(RVO LANGUAGES CXX)

include(CTest)
include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Security and optimization flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wformat=2 -Wformat-security)
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(-O3 -DNDEBUG)
    endif()
    # Security hardening
    add_compile_options(-fstack-protector-strong -D_FORTIFY_SOURCE=2)
    add_link_options(-Wl,-z,relro,-z,now)
endif()

option(RVO_BUILD_EXAMPLES "Build RVO2 examples" OFF)

add_subdirectory(src)

if(RVO_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

include(CPack)
