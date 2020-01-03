package com.rc.platform.task

import com.rc.platform.config.SpeckConfig
import com.rc.platform.config.EventMapConfig.eventMap
import com.rc.platform.task.RCAlsTrain.deleteFile
import com.rc.platform.task.RCLogisticRegressionDataProcess.deleteFile
import com.rc.platform.task.RCLogisticRegressionTrain.{deleteFile, getOffsetDay, getToday}
import com.rc.platform.task.RCPornographicDiscriminationPrediction.{getOffsetDay, getToday}
import com.rc.platform.util.RedisDao
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.ml.feature.{Bucketizer, OneHotEncoder, OneHotEncoderEstimator, QuantileDiscretizer, StringIndexer, VectorAssembler, VectorIndexer}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{udf, _}
import ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier
import org.apache.spark.sql.types.{IntegerType, StructField}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

object RCSnapShotsPrediction extends TaskSchedule {
  def main(args:Array[String]) {

    /***
     * 基础设定
     ***/

    val spark = SpeckConfig.getSpeckConfigInstance(this.getClass.getSimpleName.filter(!_.equals('$')))
    import spark.implicits._
    import spark.sql

    println("*"*80)
    println("*"*80)
    println("*"*80)

    // 最早提取截止日期
    val recordDay = "2019-12-11"
    val offset = args(0).toInt
    val offsetDay = getOffsetDay(offset)
    // 获取今日时间
    val today = getToday()
    val historyOffset = -180
    val historyOffsetDay = getOffsetDay(historyOffset)
    // 分桶数
    val quantileSize = 15
    // 模型保存路径
    val lr_model_model_path = "s3://bigdata-rc/algorithm-data/recommend/SnapshotsPrediction/model"
    // 自定义语言过滤函数
    val filterManyLanguageUDF = udf((languageId: String) => {
      var res = false
      if (languageId.split(",").length == 1) {
        res = true
      }
      if (languageId == "-1") {
        res = false
      }
      res
    })

    println("*"*80)
    println("*"*80)
    println("*"*80)

    /***
     * 特征选取部分
     ***/

    // 用户基本特征
    val userDF = sql(
      s"""SELECT user_id,
         |       app_id,
         |       gender,
         |       country_id,
         |       gold_num,
         |       language_id,
         |       platform_type,
         |       type,
         |       pay_status,
         |       eroticism_behavior,
         |       sign_eroticism,
         |       channel
         |FROM rc_video_chat.rc_user where dt>='${historyOffsetDay}'""".stripMargin).filter(filterManyLanguageUDF($"language_id")).filter($"country_id" > 0).withColumn("language_id", $"language_id".cast(IntegerType))

    val userCountryIdDF = userDF.withColumnRenamed("user_id", "target_user_id").select("target_user_id", "country_id")

    // 视频记录
    val videoRecordDF = sql(
      s"""SELECT user_id,
         |       matched_id        target_user_id,
         |       video_time / 1000 video_time,
         |       gender_condition,
         |       request_type,
         |       goddess_location,
         |       goddess_video,
         |       gender,
         |       room_id,
         |       dt
         |FROM rc_live_chat_statistics.rc_video_record
         |where dt >= '${recordDay}'""".stripMargin)

    // 活跃用户
    val aliveUserDF = sql(
      s"""SELECT user_id
         |FROM rc_live_chat_statistics.rc_user_record where dt='${recordDay}'""".stripMargin)
    val boyAliveUserDF = aliveUserDF.filter($"gender" === 1)
    val girlAliveUserDF = aliveUserDF.filter($"gender" === 2).withColumnRenamed("user_id", "target_user_id")

    val girlConfigDF = sql(s"""SELECT * FROM  rc_video_chat.rc_partner_level_price_config """.stripMargin).withColumnRenamed("id", "price_level_id_new").select("level", "price_level_id_new")
    val goddessDF = sql(s"""SELECT * FROM  rc_video_chat.rc_goddess """.stripMargin).withColumnRenamed("user_id", "target_user_id")
    val goddessUserIdDF = goddessDF.select("target_user_id")
    val normalDF = sql(s"""SELECT * FROM  rc_video_chat.rc_normal_group_user """.stripMargin).withColumnRenamed("user_id", "target_user_id")
    val normalUserIdDF = normalDF.select("target_user_id")
    val totalGirlsDF = goddessUserIdDF.union(normalUserIdDF).join(girlAliveUserDF, Seq("target_user_id")).distinct()
    val totalGirlLevel = goddessDF.select("target_user_id", "price_level_id_new")
      .union(normalDF.select("target_user_id", "price_level_id_new"))
      .join(girlAliveUserDF, Seq("target_user_id"))
      .join(girlConfigDF, Seq("price_level_id_new")).select("target_user_id", "level")
      .withColumnRenamed("level", "girl_level_category")
      .distinct()
    val girlCountryIdMap = totalGirlsDF.join(userCountryIdDF, "target_user_id").as[(Int, Int)].collect().toMap
//    val bcGirlCountryIdMap = spark.sparkContext.broadcast(girlCountryIdMap)

    val boyProfileNames = Seq(
      "user_id",
      "boy_app_id_category",
      "boy_gender_category",
      "boy_country_id_category",
      "boy_gold_num_continuous",
      "boy_language_id_category",
      "boy_platform_type_category",
      "boy_type_category",
      "boy_pay_status_category",
      "boy_eroticism_behavior_category",
      "boy_sign_eroticism_category",
      "boy_channel_category")
    val girlProfileNames = Seq(
      "target_user_id",
      "girl_app_id_category",
      "girl_gender_category",
      "girl_country_id_category",
      "girl_gold_num_continuous",
      "girl_language_id_category",
      "girl_platform_type_category",
      "girl_type_category",
      "girl_pay_status_category",
      "girl_eroticism_behavior_category",
      "girl_sign_eroticism_category",
      "girl_channel_category")

    val boyProfilePath = (offset to -1).map("s3://bigdata-rc/algorithm-data/recommend/LogisticRegression/features/boy/" + getOffsetDay(_))
    val girlProfilePath = (offset to -1).map("s3://bigdata-rc/algorithm-data/recommend/LogisticRegression/features/girl/" + getOffsetDay(_))
    val boyToGirlProfilePath = (offset to -1).map("s3://bigdata-rc/algorithm-data/recommend/LogisticRegression/features/boyToGirl/" + getOffsetDay(_))
    val convertUDF = udf((array: Seq[Float]) => {
      Vectors.dense(array.toArray.map(_.toDouble))
    })
    val totalUserAlsDF = spark.sqlContext.read.parquet("s3://bigdata-rc/algorithm-data/recommend/ALS/userFeatures/" + today).withColumnRenamed("features", "boy_als_features").withColumn("boy_als_features", convertUDF($"boy_als_features"))
    val totalGirlAlsDF = spark.sqlContext.read.parquet("s3://bigdata-rc/algorithm-data/recommend/ALS/productFeatures/" + today)
      .withColumnRenamed("features", "girl_als_features")
      .withColumn("girl_als_features", convertUDF($"girl_als_features"))

    val boyProfileDF = spark.sqlContext.read.parquet(boyProfilePath: _*)
    val girlProfileDF = spark.sqlContext.read.parquet(girlProfilePath: _*)
    val boyToGirlProfileDF = spark.sqlContext.read.parquet(boyToGirlProfilePath: _*)

    println("*"*80)
    println("*"*80)
    println("*"*80)
    println("文件读取完成")
    println("*"*80)
    println("*"*80)
    println("*"*80)



    /***
     * 产生label部分
     ***/

    // 将按照固定3、6秒截图的用户与5%均匀截图的测试用户区分开
    val chosenUserDF = sql("select user_id, location, room_id from rc_live_chat_statistics.rc_video_snapshots where dt>'2019-12-11'").groupBy("user_id").agg(min("location").as("min"), max("location").as("max")).filter($"max">6).filter($"min"===5).select("user_id")

    // 统计5%用户的色情截图时间
    val shotsDF = sql("select user_id, location, room_id from rc_live_chat_statistics.rc_video_snapshots where dt>'2019-12-11' and violations_label=0").groupBy("user_id").agg(collect_list("location").as("collections")).join(chosenUserDF,Seq("user_id"),"inner").na.fill(0)

    // 自定义函数，取出数组中出现频次最高的数字
    val sortingUDF = udf((collections:Seq[Int]) => {
      val clo = collections.toList.map((_,1)).groupBy(_._1).mapValues(_.size).toList.sortBy(-_._2)
      val res = clo(0)._1
      res
    })

    // 获取每个user_id对应的出现频次最高的色情截图时间
    val statisticsDF = shotsDF.withColumn("most",sortingUDF($"collections"))

    // 对截图时间区间进行分桶，给出label标签,并进行升采样
    val splits = Array(0,6,11,21,31,46,61,101, Double.PositiveInfinity)
    val bucketizerTransformers = new Bucketizer().setInputCol("most").setOutputCol("label").setSplits(splits).setHandleInvalid("skip")
    val labelDF = bucketizerTransformers.transform(statisticsDF).select( "user_id","label")

    /***
     * 特征合并部分
     ***/

    val boySumProfileDF = boyProfileDF.groupBy("user_id").sum(boyProfileDF.columns.filter(_ != "user_id"): _*)
    val girlSumProfileDF = girlProfileDF.groupBy("target_user_id").sum(girlProfileDF.columns.filter(_ != "target_user_id"): _*)

    def fillNaWithVectorUDF = udf((vector: Vector, size: Int) => {
      val fillArray = Array.fill(size)(0.0d)
      var res = Vectors.dense(fillArray)
      if (vector.isInstanceOf[Vector]) {
        res = vector
      }
      res
    })

    // 完整特征合并
    val resDF = labelDF
      .join(userDF.toDF(boyProfileNames: _*), Seq("user_id"))
      .join(boySumProfileDF, Seq("user_id"), "left_outer")
      .join(totalUserAlsDF, Seq("user_id"), "left_outer")
      .withColumn("boy_als_features", fillNaWithVectorUDF($"boy_als_features", lit(10))).na.fill(0).sample(true,10)

    println("*"*80)
    println("*"*80)
    println("*"*80)
    println("resDF 合并完成")
    println(resDF.count())
    println("*"*80)
    println("*"*80)
    println("*"*80)

    /***
     * 特征工程部分
     ***/

    // 特征组装
    val vectorNames = resDF.columns.filterNot(_.contains("user_id")).filterNot(_.contains("label"))
    val vectorAssembler = new VectorAssembler().setInputCols(vectorNames).setOutputCol("featuresOut")

    val vectorAssemblerDF = vectorAssembler.transform(resDF).drop(vectorNames: _*)

    // VectorIndexer对连续值进行分桶
    val featureIndexer = new VectorIndexer().setInputCol("featuresOut").setOutputCol("indexedFeatures").setMaxCategories(quantileSize).fit(vectorAssemblerDF)
    val df2 = featureIndexer.transform(vectorAssemblerDF)
    val data = df2.select("user_id",  "indexedFeatures", "label").cache()

    // 划分训练集和测试集
    val Array(trainingDF, testDF) = data.randomSplit(Array(0.7, 0.3))


    /***
     * 模型训练
     ***/

    // 构建XGBoost模型框架和参数
    val xgbParam = Map("eta" -> 0.1f,
      "missing" -> 0,
      "objective" -> "multi:softprob",
      "num_class" -> 8,
      "max_depth" -> 28,
      "gamma" -> 3,
      "num_round" -> 5,
      "silent" -> 0,
      "num_workers" -> 2)

    val xgbClassifier = new XGBoostClassifier(xgbParam).setFeaturesCol("indexedFeatures").setLabelCol("label")

    // 训练并测试
    val model = xgbClassifier.fit(trainingDF)
    val test = model.transform(testDF)
//    val train = model.transform(trainingDF)

    // 模型评估 MulticlassClassificationEvaluator
    val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy").setPredictionCol("prediction").setLabelCol("label")
    val classificationMetric = evaluator.evaluate(test)
//    val classificationMetric = evaluator.evaluate(train)

    // 打印测试结果
    println("*"*80)
    println("*"*80)
    println("*"*80)
    println(s"${evaluator.getMetricName} = $classificationMetric" +
      "\ncolumn count: " + data.head().getAs[Vector]("indexedFeatures").size +
      "\nfinal data label 0 count: " + data.filter(col("label") === 0).count() +
      "\nfinal data label 1 count: " + data.filter(col("label") === 1).count() +
      "\nfinal data label 2 count: " + data.filter(col("label") === 2).count() +
      "\nfinal data label 3 count: " + data.filter(col("label") === 3).count() +
      "\nfinal data label 4 count: " + data.filter(col("label") === 4).count() +
      "\nfinal data label 5 count: " + data.filter(col("label") === 5).count() +
      "\nfinal data label 6 count: " + data.filter(col("label") === 6).count() +
      "\nfinal data label 7 count: " + data.filter(col("label") === 7).count()
    )
    println("*"*80)
    println("*"*80)
    println("*"*80)

    /***
     * 模型持久化
     ***/

    // deleteFile(spark, lr_model_model_path, today, historyOffsetDay)
    model.write.overwrite().save(lr_model_model_path + today)

    println("*"*80)
    println("*"*80)
    println("*"*80)
    println(s"Model has been saved to ${lr_model_model_path + today} !")


  }
}
