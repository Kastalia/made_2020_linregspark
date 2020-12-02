package org.apache.spark.ml.made

import breeze.linalg._
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.util._
import org.apache.spark.ml.linalg.{DenseVector, Vector, VectorUDT, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasInputCol, HasLabelCol, HasMaxIter, HasOutputCol, HasPredictionCol, HasStepSize}
import org.apache.spark.mllib
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.functions.lit
import org.apache.spark.sql.types.{StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Encoder, Row}

trait LinearRegressionParams extends HasLabelCol with HasFeaturesCol with HasPredictionCol {
  def setLabelCol(value: String): this.type = set(labelCol, value)
  def setFeaturesCol(value: String): this.type = set(featuresCol, value)
  def setPredictionCol(value: String): this.type = set(predictionCol, value)

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, $(featuresCol), new VectorUDT())
    SchemaUtils.checkNumericType(schema, $(labelCol))

    if (schema.fieldNames.contains($(predictionCol))) {
      SchemaUtils.checkColumnType(schema, getPredictionCol, new VectorUDT())
    } else {
      SchemaUtils.appendColumn(schema, StructField(getPredictionCol, new VectorUDT()))
    }
    schema
  }
}


class LinearRegression(override val uid: String, maxIter: Int, stepSize: Double) extends Estimator[LinearRegressionModel] with LinearRegressionParams
  with DefaultParamsWritable {

  def this(maxIter: Int, stepSize: Double) = this(Identifiable.randomUID("linearRegression"), maxIter, stepSize)

  private case class Params(maxIter: Int, stepSize: Double)
  //private[made] def this (uid: String) = this(uid, 1000, 0.1)

  override def fit(dataset: Dataset[_]): LinearRegressionModel = {
    implicit val encoder: Encoder[Vector] = ExpressionEncoder()
    val n_features = MetadataUtils.getNumFeatures(dataset, $(featuresCol))
    var coefficients = breeze.linalg.DenseVector.rand[Double](n_features + 1)

    val assembler = new VectorAssembler()
      .setInputCols(Array("intercept", $(featuresCol), $(labelCol)))
      .setOutputCol("features_assembler")
    val features = assembler
      .transform(dataset.withColumn("intercept", lit(1)))
      .select("features_assembler").as[Vector]

    for (_ <- 0 to maxIter) {
      val loss = features.rdd.mapPartitions((data: Iterator[Vector]) => {
        val summarizer = new MultivariateOnlineSummarizer()
        data.foreach(v => {
          val X = v.asBreeze(0 until coefficients.size).toDenseVector
          val y = v.asBreeze(-1)
          val grads = X * (breeze.linalg.sum(X * coefficients) - y)
          summarizer.add(mllib.linalg.Vectors.fromBreeze(grads))
        })
        Iterator(summarizer)
      }).reduce(_ merge _)
      coefficients = coefficients - stepSize * loss.mean.asBreeze
    }

    copyValues(new LinearRegressionModel(
      Vectors.fromBreeze(coefficients(1 until coefficients.size)).toDense,
      coefficients(0))
    ).setParent(this)
  }

  override def copy(extra: ParamMap): Estimator[LinearRegressionModel] = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)
}

object LinearRegression extends DefaultParamsReadable[LinearRegression]


class LinearRegressionModel private[made](
                                           override val uid: String,
                                           val coefficients: DenseVector,
                                           val intercept: Double) extends Model[LinearRegressionModel]
  with LinearRegressionParams with MLWritable {

  private[made] def this(coefficients: DenseVector, intercept: Double) =
    this(Identifiable.randomUID("linearRegressionModel"), coefficients, intercept)

  override def copy(extra: ParamMap): LinearRegressionModel = copyValues(
    new LinearRegressionModel(uid, coefficients, intercept))

  override def transform(dataset: Dataset[_]): DataFrame = {
    val transformUdf = dataset.sqlContext.udf.register(uid + "_predict",
      (x: Vector) => {
        Vectors.fromBreeze(breeze.linalg.DenseVector(coefficients.asBreeze.dot(x.asBreeze) + intercept))
      })

    dataset.withColumn($(predictionCol), transformUdf(dataset($(featuresCol))))
  }

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

  override def write: MLWriter = new DefaultParamsWriter(this) {
    override protected def saveImpl(path: String): Unit = {
      super.saveImpl(path)

      sqlContext.createDataFrame(Seq(coefficients -> intercept)).write.parquet(path + "/vectors")
    }
  }
}

object LinearRegressionModel extends MLReadable[LinearRegressionModel] {
  override def read: MLReader[LinearRegressionModel] = new MLReader[LinearRegressionModel] {
    override def load(path: String): LinearRegressionModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc)
      val vectors = sqlContext.read.parquet(path + "/vectors")
      implicit val encoder: Encoder[Vector] = ExpressionEncoder()

      val coefficients = vectors.select(vectors("_1").as[Vector]).first()
      val intercept = vectors.select(vectors("_2")).first().getDouble(0)

      val model = new LinearRegressionModel(coefficients.toDense, intercept)
      metadata.getAndSetParams(model)
      model
    }
  }
}