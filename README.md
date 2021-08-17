# Spark Matrix Multiplication: A Short Dive into Parallellism in Apache Spark
<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project
An example demonstrating matrix multiplication using Apache Spark in Python, created for the purpooses of an oral presentation in CAB401 High Performance and Parallel Computing at the Queensland University of Technology (QUT), Semester 2, 2021. This project is a simple Python script that compares the performance of Apache Spark, i.e. `pyspark` against standard matrix multiplication implementation.

### Built With
This script relies on the following dependencies.

* [Scala](https://www.scala-lang.org)
* [Java](https://www.java.com/en/)
* [Python3](https://www.python.org)
* [Apache Spark](https://spark.apache.org)
* [SciPy](https://www.scipy.org)

<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

This project has been built and run targetting a Unix/Linux environment. The steps used to instantiate and setup the environment for the use of this script may vary depending on operating system. For the purposes of this it is assumed that you have `homebrew` and `python3` installed.

### Installation

1. Install Java to the local machine, ```brew install java```
2. Install Scala to the local machine, ```brew install scala```
3. Install Apache Spark, ```brew install spark```
4. Install `scipy`, `pyspark`, and `findspark` in Python, i.e. ```pip3 install pyspark```
5. Clone the repo
   ```sh
   git clone https://github.com/jamestkelly/spark-matrix-multiplication.git
   ```
6. Run the script from the command-line with ```python3 path/to/file.py```

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<!-- CONTACT -->
## Contact

Jim Tran Kelly

* Student Number: N9763686
* General Email: [jimkelly.t@outlook.com](jimkelly.t@outlook.com)
* University Contact: [jim.kelly@connect.qut.edu.au](jim.kelly@connect.qut.edu.au)

Project Link: [https://github.com/jamestkelly/titanicSpark](https://github.com/jamestkelly/titanicSpark/tree/master)

<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
* [How to Convert a DistributedMatrix to Scipy Sparse or Numpy Array](https://stackoverflow.com/questions/54083978/convert-a-distributedmatrix-to-scipy-sparse-or-numpy-array)
* [How to Multiply Two Numpy Matrices in PySpark](https://stackoverflow.com/questions/42889965/multiply-two-numpy-matrices-in-pyspark)