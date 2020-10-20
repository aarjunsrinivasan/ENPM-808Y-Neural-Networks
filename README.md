## ARoBot - A PID controller Interface module for Ackerman Steering based Autonmous Robots
[![Build Status](https://travis-ci.org/rajesh1996/ARoBot-MPC-Controller-Interface.svg?branch=master)](https://travis-ci.org/rajesh1996/ARoBot-MPC-Controller-Interface)
[![Coverage Status](https://coveralls.io/repos/github/rajesh1996/ARoBot-MPC-Controller-Interface/badge.svg?branch=master)](https://coveralls.io/github/rajesh1996/ARoBot-MPC-Controller-Interface?branch=master)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
---

## Authors
* Rajeshwar N S  - Sprint 1 (Driver) Sprint 2 (Navigator)
* Arjun Srinivasan Ambalam - Sprint 1 (Navigator) Sprint 2 (Driver)

## Overview
A controller interface for an Ackerman Kinematic model which continiuosly predicts the steering angle and wheel velocities

## Agile Iterative Process Log sheet

[Agile log sheet](https://drive.google.com/file/d/16rTux2BRGNPGFXxO9nNet1ViL05aODM7/view?usp=sharing)

## Dependencies
1. C++ 11/14/17
2. gnuplot- http://stahlke.org/dan/gnuplot-iostream/
<br>Install gnuplot
```
sudo apt-get install gnuplot
```

## Standard install via command-line
```
git clone --recursive https://github.com/rajesh1996/ARoBot-MPC-Controller-Interface 
cd <path to repository>
mkdir build
cd build
cmake ..
make
Run tests: ./test/cpp-test
Run program: ./app/shell-app
```

## License
This software has been lincesed under [BSD 3-Clause](https://github.com/rajesh1996/ARoBot-MPC-Controller-Interface/blob/master/LICENSE.md)

## Building for code coverage
```
sudo apt-get install lcov
cmake -D COVERAGE=ON -D CMAKE_BUILD_TYPE=Debug ../
make
make code_coverage
```
This generates a index.html page in the build/coverage sub-directory that can be viewed locally in a web browser.

## Run cppcheck

To detect bugs and perform static code analysis, cppcheck is used. It can be installed with the following command:
```
sudo apt-get install cppcheck
```
To run cppcheck, run the following command in the `/PID-Controller-TDD` folder:
```
cppcheck --enable=all --std=c++11 -I include/ --suppress=missingIncludeSystem $( find . -name *.cpp | grep -vE -e "^./build/" -e "^./vendor/" )
```

## Run cpplint

To check if the Google style guide is followed cpplint is used:
```
sudo apt-get install cpplint
```

To run the cpplint, follow the following commands in the `/PID-Controller-TDD` directory:
```
cpplint $( find . -name \*.hpp -or -name \*.cpp | grep -vE -e "^./build/" -e "^./vendor/" -e "^./data" )



	


