package org.apache.spark.ml.made

import com.google.common.io.Files
import org.scalatest._
import flatspec._
import matchers._

import breeze.linalg.{*, DenseMatrix, DenseVector}
import breeze.numerics._
import breeze.stats.mean

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.{Pipeline, linalg, PipelineModel}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.{DataFrame}


class LinearRegressionTest extends AnyFlatSpec with should.Matchers with WithSpark {

  lazy val data: DataFrame = LinearRegressionTest._data
  lazy val coefficients: linalg.DenseVector = LinearRegressionTest._coefficients
  lazy val intercept: Double = LinearRegressionTest._intercept
  lazy val X: DenseMatrix[Double] = LinearRegressionTest._X
  lazy val y: DenseVector[Double] = LinearRegressionTest._y

  val stepSize = 1
  val maxIter = 100
  val delta = 1e-1

  "Estimator" should "test train" in {
    val estimator: LinearRegression = new LinearRegression(maxIter=maxIter, stepSize=stepSize)
      .setFeaturesCol("features")
      .setLabelCol("y")
      .setPredictionCol("prediction")

    val model = estimator.fit(data)

    model.intercept should be(intercept +- delta)
    model.coefficients(0) should be(coefficients(0) +- delta)
    model.coefficients(1) should be(coefficients(1) +- delta)
    model.coefficients(2) should be(coefficients(2) +- delta)
  }

  "Model" should "test predict" in {
    val model: LinearRegressionModel = new LinearRegressionModel(
      coefficients,
      intercept
    ).setFeaturesCol("features")
      .setLabelCol("y")
      .setPredictionCol("prediction")

    val pred = DenseVector(model.transform(data).select("prediction").collect().map(x => x.getAs[Vector](0)(0)))

    sqrt(mean(pow(pred - y, 2))) should be(0.0 +- delta)
  }
}


object LinearRegressionTest extends WithSpark {
  lazy val _noise: DenseVector[Double] = DenseVector.rand[Double](100000) * 0.03
  lazy val _X: DenseMatrix[Double] = DenseMatrix.rand[Double](100000, 3)
  lazy val _coefficients: linalg.DenseVector = Vectors.dense(-1.0, 2.0, -3).toDense
  lazy val _intercept: Double = 0.9
  lazy val _y: DenseVector[Double] = _X * _coefficients.asBreeze + _intercept + _noise

  lazy val _data: DataFrame = {
    import sqlc.implicits._

    val tmp = DenseMatrix.horzcat(_X, _y.asDenseMatrix.t)
    val df = tmp(*, ::).iterator
      .map(x => (x(0), x(1), x(2), x(3)))
      .toSeq
      .toDF("x1", "x2", "x3", "y")

    val assembler = new VectorAssembler()
      .setInputCols(Array("x1", "x2", "x3"))
      .setOutputCol("features")
    val output = assembler.transform(df).select("features", "y")

    output
  }
}
