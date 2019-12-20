
package com.rc.platform.task

import com.rc.platform.config.SpeckConfig
import com.rc.platform.config.EventMapConfig.eventMap
import com.rc.platform.task.RCAlsTrain.deleteFile
import com.rc.platform.task.RCLogisticRegressionDataProcess.deleteFile
import com.rc.platform.task.RCLogisticRegressionTrain.{deleteFile, getOffsetDay, getToday}
import com.rc.platform.util.RedisDao
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.ml.feature.{OneHotEncoder, OneHotEncoderEstimator, QuantileDiscretizer, StringIndexer, VectorAssembler, VectorIndexer}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{udf, _}
import org.apache.spark.sql.types.IntegerType

object RCPornographicTolerance extends TaskSchedule{
  def main(args:Array[String]){

    /***
     * 基础设定
     ***/

    val spark = SpeckConfig.getSpeckConfigInstance(this.getClass.getSimpleName.filter(!_.equals('$')))
    import spark.implicits._
    import spark.sql

    // 最早提取截止日期
    //    val offsetDay = "2019-12-01"
    val recordDay = "2019-12-16"
    val offset = args(0).toInt
    val offsetDay = getOffsetDay(offset)
    // 获取今日时间
    val today = getToday()
    val historyOffset = -180
    val historyOffsetDay = getOffsetDay(historyOffset)
    // 分桶数
    val quantileSize = 10
    // 模型保存路径
    val lr_model_model_path = "s3://bigdata-rc/algorithm-data/recommend/RCPornographicTolerance/model"
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

    /***
     * 特征工程部分
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
         |FROM rc_video_chat.rc_user where dt>='${historyOffsetDay}'""".stripMargin)
      .filter(filterManyLanguageUDF($"language_id"))
      .filter($"country_id" > 0)
      .withColumn("language_id", $"language_id".cast(IntegerType))

    val userCountryIdDF = userDF.withColumnRenamed("user_id", "target_user_id")
      .select("target_user_id", "country_id")

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
    //    val boyAliveUserDF = aliveUserDF.filter($"gender" === 1)
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
    val bcGirlCountryIdMap = spark.sparkContext.broadcast(girlCountryIdMap)

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

    //    val boyProfilePath = (offset to -1).map("s3://bigdata-rc/algorithm-data/recommend/LogisticRegression/features/boy/" + getOffsetDay(_))
    val girlProfilePath = (offset to -1).map("s3://bigdata-rc/algorithm-data/recommend/LogisticRegression/features/girl/" + getOffsetDay(_))
    //    val boyToGirlProfilePath = (offset to -1).map("s3://bigdata-rc/algorithm-data/recommend/LogisticRegression/features/boyToGirl/" + getOffsetDay(_))
    val convertUDF = udf((array: Seq[Float]) => {
      Vectors.dense(array.toArray.map(_.toDouble))
    })
    //    val totalUserAlsDF = spark.sqlContext.read.parquet("s3://bigdata-rc/algorithm-data/recommend/ALS/userFeatures/" + today)
    //      .withColumnRenamed("features", "boy_als_features")
    //      .withColumn("boy_als_features", convertUDF($"boy_als_features"))
    val totalGirlAlsDF = spark.sqlContext.read.parquet("s3://bigdata-rc/algorithm-data/recommend/ALS/productFeatures/" + today)
      .withColumnRenamed("features", "girl_als_features")
      .withColumn("girl_als_features", convertUDF($"girl_als_features"))

    //    val boyProfileDF = spark.sqlContext.read.parquet(boyProfilePath: _*)
    val girlProfileDF = spark.sqlContext.read.parquet(girlProfilePath: _*)
    //    val boyToGirlProfileDF = spark.sqlContext.read.parquet(boyToGirlProfilePath: _*)

    /***
     * 产生标签
     ***/

    //    val normalDF = snapShotsDF.filter($"violations_label" === 2)
    //      .select("room_id").groupBy("room_id").agg(count("room_id").as("normal_count"))


    // snapShots 与 videoRecord基于room_id相关联, 并分别统计截图中色情/性感/正常的数量
    //    val vioDF = snapShotsDF.join(videoRecordDF,"room_id")
    //      .select("user_id","target_user_id").groupBy("user_id","target_user_id").agg(count("user_id").as("count"))

    //    val sexualDF = snapShotsDF.join(videoRecordDF,"room_id").filter($"violations_label" === 1)
    //      .select("user_id","target_user_id").groupBy("user_id","target_user_id")
    //      .agg(count("user_id").as("sexual_count"))

    //    val normalDF = snapShotsDF.join(videoRecordDF,"room_id").filter($"violations_label" === 2)
    //      .select("user_id","target_user_id").groupBy("user_id","target_user_id")
    //      .agg(count("user_id").as("normal_count"))

    // 分别统计截图中色情/性感/正常的数量
    //    val violationDF = snapShotsDF.join(aliveUserDF,"user_id").filter($"violations_label" === 0)
    //      .select("user_id").groupBy("user_id").agg(count("user_id").as("violations_count"))
    //    val sexualDF = snapShotsDF.join(aliveUserDF,"user_id").filter($"violations_label" === 1)
    //      .select("user_id").groupBy("user_id").agg(count("user_id").as("sexual_count"))
    //    val normalDF = snapShotsDF.join(aliveUserDF,"user_id").filter($"violations_label" === 2)
    //      .select("user_id").groupBy("user_id").agg(count("user_id").as("normal_count"))

    // 对截图记录中的场景、视频类型、匹配model进行统计
    //    val pageDF = snapShotsDF.join(aliveUserDF,"user_id")
    //      .select("user_id","page").groupBy("user_id")
    //      .agg(max("page").as("page_sum"))
    //    val modelDF = snapShotsDF.join(aliveUserDF,"user_id")
    //      .select("user_id","model").groupBy("user_id")
    //      .agg(max("model").as("model_sum"))
    //    val videoTypeDF = snapShotsDF.join(aliveUserDF,"user_id")
    //      .select("user_id","video_type").groupBy("user_id")
    //      .agg(max("video_type").as("video_type_sum"))


    //合并截图统计结果
    //    val snapShotsUnionDF = snapShotsDF.join(violationDF,Seq("user_id"),"left_outer")
    //      .join(sexualDF,Seq("user_id"),"left_outer")
    //      .join(normalDF,Seq("user_id"),"left_outer")
    //      .join(pageDF,Seq("user_id"),"left_outer")
    //      .join(modelDF,Seq("user_id"),"left_outer")
    //      .join(videoTypeDF,Seq("user_id"),"left_outer")

    // 近期视频截图记录
    val snapShotsDF = sql(
      s"""SELECT room_id,
         |user_id        target_user_id,
         |location,
         |violations_label,
         |model
         |FROM rc_live_chat_statistics.rc_video_snapshots where dt>='${recordDay}'
              """.stripMargin)

    // 每个room_id对应的色情标签最小值，0：色情，1：性感，2：正常
    val roomDF = snapShotsDF.select("room_id","violations_label")
      .groupBy("room_id").agg(min("violations_label").as("violations"))
      .na.drop()

    // 分别统计每个用户名下的截图中色情/性感/正常的数量
    val violationDF = snapShotsDF.join(girlAliveUserDF,Seq("target_user_id"),"left_outer").filter($"violations_label" === 0)
      .select("target_user_id").groupBy("target_user_id").agg(count("target_user_id").as("violations_continuous"))
    val sexualDF = snapShotsDF.join(girlAliveUserDF,Seq("target_user_id"),"left_outer").filter($"violations_label" === 1)
      .select("target_user_id").groupBy("target_user_id").agg(count("target_user_id").as("sexual_continuous"))
    val otherDF = snapShotsDF.join(girlAliveUserDF,Seq("target_user_id"),"left_outer").filter($"violations_label" === 2)
      .select("target_user_id").groupBy("target_user_id").agg(count("target_user_id").as("normal_continuous"))

    // 筛选除所有时长大于20秒的视频，并选取其对应的target_user_id和violations
    val allVideoViolationDF = videoRecordDF.join(roomDF,Seq("room_id"),"left_outer").distinct()
      .filter($"video_time" >= 20).select("target_user_id","room_id","violations")

    //  统计用户名下所有时长大于20秒的违规的视频数和总视频数
    val longVideoViolationDF = allVideoViolationDF.filter($"violations" === 0).select("target_user_id")
      .groupBy("target_user_id").agg(count("target_user_id").as("violation_video_sum"))
    val totalVideoViolationDF = allVideoViolationDF.select("target_user_id")
      .groupBy("target_user_id").agg(count("target_user_id").as("total_video_sum"))

    // 产生标签函数
    val getLabelUDF = udf((violation_video_sum:Int,total_video_sum:Int) =>
    {
      var res = -1
      if(violation_video_sum/total_video_sum > 0.4){res = 1}else{res = 0}
      res
    })

    // 产生标签
    val originLabelDF = longVideoViolationDF.join(totalVideoViolationDF,Seq("target_user_id"),"left_outer")
      .withColumn("label", getLabelUDF($"sexual_video_sum",$"total_video_sum"))
      .filter($"label" =!= -1)
      .groupBy("target_user_id").agg(max("label").as("label"))
      .na.fill(0)

    // 对负样本进行欠采样
    val negativeDF = originLabelDF.filter($"label" === 0).sample(true,0.01)
    val positiveDF = originLabelDF.filter($"label" === 1)
    val labelDF = negativeDF.union(positiveDF)

    println("*"*80)
    println("*"*80)
    println("*"*80)
    println("标签labelDF 生成！")
    println("*"*80)
    println("*"*80)
    println("*"*80)

    /***
     * 特征合并部分
     ***/

    // 合并用户画像以及男对女行为
    //    val boySumProfileDF = boyProfileDF.groupBy("user_id").sum(boyProfileDF.columns.filter(_ != "user_id"): _*)
    //      .join(violationDF,Seq("user_id"),"left_outer")
    //      .join(sexualDF,Seq("user_id"),"left_outer")
    //      .join(otherDF,Seq("user_id"),"left_outer")
    val girlSumProfileDF = girlProfileDF.groupBy("target_user_id").sum(girlProfileDF.columns.filter(_ != "target_user_id"): _*)
      .join(violationDF,Seq("target_user_id"),"left_outer")
      .join(sexualDF,Seq("target_user_id"),"left_outer")
      .join(otherDF,Seq("target_user_id"),"left_outer")
    //    val boyToGirlSumProfileDF = boyToGirlProfileDF.groupBy("user_id", "target_user_id").sum(boyToGirlProfileDF.columns.filter(!_.contains("user_id")): _*)

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
      //      .join(boyToGirlSumProfileDF, Seq("user_id", "target_user_id"), "left_outer")
      //      .join(userDF.toDF(boyProfileNames: _*), Seq("user_id"))
      //      .join(boySumProfileDF, Seq("user_id"), "left_outer")
      //      .join(totalUserAlsDF, Seq("user_id"), "left_outer")
      //      .withColumn("boy_als_features", fillNaWithVectorUDF($"boy_als_features", lit(10)))
      .join(girlSumProfileDF, Seq("target_user_id"), "left_outer")
      .join(totalGirlAlsDF, Seq("target_user_id"), "left_outer")
      .withColumn("girl_als_features", fillNaWithVectorUDF($"girl_als_features", lit(10)))
      .join(totalGirlLevel, Seq("target_user_id"), "left_outer")
      .join(userDF.toDF(girlProfileNames: _*), Seq("target_user_id"))
      .na.fill(0)

    println("*"*30)
    println("resDF 合并完成")
    println(resDF.count())
    println("*"*30)

    // 字符类别特征转离散特征, 此类处理的特征以 _idx结尾
    val stringIndexerTransformers = resDF.columns.filter(_.contains("category")).flatMap((columnName: String) => {
      val stringIndexer = new StringIndexer()
        .setInputCol(columnName)
        .setOutputCol(s"${columnName}_idx")
        .setHandleInvalid("skip")
      Array(stringIndexer)
    })

    // 连续值分桶成离散类别, 此类处理的特征以 _quantiles结尾
    val inputColumnsNames = resDF.columns.filter {
      _.contains("continuous")
    }
    val quantileOutputColumnsNames = inputColumnsNames.map(_ + "_quantile")
    val quantileTransformers = new QuantileDiscretizer()
      .setInputCols(inputColumnsNames)
      .setOutputCols(quantileOutputColumnsNames)
      .setNumBuckets(quantileSize)
      .setRelativeError(0.1)
      .setHandleInvalid("skip")

    // 构建pipeline
    val featureTransformDF = new Pipeline()
      .setStages(stringIndexerTransformers ++ Array(quantileTransformers))
      .fit(resDF).transform(resDF).drop(inputColumnsNames: _*) // :_* 将Array作为多参数传入

    // 独热编码--将离散的类别特征转为离散向量
    val oneHotInputColumnsNames = featureTransformDF.columns.filter { name => name.contains("_quantile") || name.contains("_idx") }
    val oneHotOutputColumnsNames = oneHotInputColumnsNames.map(_ + "_onehot")
    val oneHotEncoder = new OneHotEncoderEstimator()
      .setInputCols(oneHotInputColumnsNames)
      .setOutputCols(oneHotOutputColumnsNames)
      .setDropLast(false)
    val encodedDF = oneHotEncoder
      .fit(featureTransformDF)
      .transform(featureTransformDF)
      .drop(oneHotInputColumnsNames: _*)
      .na.fill(0)

    //    // 向量转规范的离散特征
    //    val featureIndexer = new VectorIndexer()
    //      .setInputCol("featuresOut")
    //      .setOutputCol("indexedFeatures")
    //      .setMaxCategories(quantileSize)
    //      .fit(vectorAssemblerDF)

    // 划分features
    //    val vectorNames = encodedDF.columns.filterNot(_.contains("user_id")).filterNot(_.contains("label")).filterNot(_.contains("target_user_id"))
    val vectorInputColumnsNames = oneHotOutputColumnsNames++Array("girl_als_features")
    //    val vectorInputColumnsNames = encodedDF.columns.filterNot(_.contains("user_id")).filterNot(_.contains("label")).filterNot(_.contains("target_user_id"))
    val vectorAssembler = new VectorAssembler()
      .setInputCols(vectorInputColumnsNames)
      .setOutputCol("featuresOut")

    //    val vectorAssemblerDF = vectorAssembler.setHandleInvalid("keep").transform(resDF)
    //    vectorAssemblerDF.printSchema()

    val data = vectorAssembler.transform(encodedDF).drop(vectorInputColumnsNames: _*)
      .select( "target_user_id","featuresOut", "label").cache()

    println("*"*30)
    println("data 合并完成")
    println("*"*30)

    /***
     * 模型训练
     ***/

    // 构建模型框架
    val lr = new LogisticRegression()
      .setMaxIter(300) //最大迭代次数
      .setRegParam(0.3) //正则化项系数
      .setElasticNetParam(0.0) //L2范数
      .setStandardization(true) //将feature标准化
      .setLabelCol("label")
      .setFeaturesCol("featuresOut")

    // 将数据进行拆分。分为训练集和测试集
    val Array(trainDF, testDF) = data.randomSplit(Array(0.1, 0.9))

    // 训练并测试
    val model = lr.fit(trainDF)
    val test = model.transform(testDF)

    // 模型评估BinaryClassificationEvaluator
    val binaryClassificationEvaluator = new BinaryClassificationEvaluator()
      .setMetricName("areaUnderROC")
      .setRawPredictionCol("rawPrediction")
      .setLabelCol("label")
    val classificationMetric = binaryClassificationEvaluator.evaluate(test)

    // 打印测试结果
    println("*"*30)
    println(s"${binaryClassificationEvaluator.getMetricName} = $classificationMetric" +
      "\ncolumn count: " + data.head().getAs[Vector]("featuresOut").size +
      "\nfinal data label 0 count: " + data.filter(col("label") === 0).count() +
      "\nfinal data label 1 count: " + data.filter(col("label") === 1).count()
    )
    println("*"*30)

    /***
     * 模型持久化
     ***/

    // deleteFile(spark, lr_model_model_path, today, historyOffsetDay)
    model.save(lr_model_model_path + today)
    println(s"Model has been saved to ${lr_model_model_path + today} !")

  }
}
