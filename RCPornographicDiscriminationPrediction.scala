
package com.rc.platform.task

import com.rc.platform.config.SpeckConfig
import com.rc.platform.config.EventMapConfig.eventMap
import com.rc.platform.task.RCAlsTrain.deleteFile
import com.rc.platform.task.RCLogisticRegressionTrain.deleteFile
import com.rc.platform.util.RedisDao
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.ml.feature.{OneHotEncoder, OneHotEncoderEstimator, QuantileDiscretizer, StringIndexer, VectorAssembler, VectorIndexer}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{udf, _}
import org.apache.spark.sql.types.IntegerType

object RCPornographicDiscriminationPrediction extends TaskSchedule{
  def main(args:Array[String]){

    /***
     * 基础设定
     ***/

    val spark = SpeckConfig.getSpeckConfigInstance(this.getClass.getSimpleName.filter(!_.equals('$')))
    import spark.implicits._
    import spark.sql

    // 最早提取截止日期
    val offsetDay = "2019-12-01"
    // 获取今日时间
    val today = getToday()
    // 分桶数
    val quantileSize = 15
    // 模型保存路径
    val lr_model_savepath = "/user/spark/recommend/PornographicDiscriminationPrediction/model"
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

    // 获取用户基本信息,14 dimensions
    val userDF = sql(
      s"""SELECT user_id,
         |       app_id,
         |       gender,
         |       age,
         |       country_id,
         |       gold_num,
         |       platform_type,
         |       type,
         |       pay_status,
         |       status,
         |       eroticism_behavior,
         |       sign_eroticism,
         |       channel
         |FROM rc_video_chat.rc_user""".stripMargin)
      .filter(filterManyLanguageUDF($"language_id"))
      .filter($"country_id" > 0)

    // 男性用户信息列表名称
    val boyNames = Seq("user_id", "boy_app_id", "boy_gender", "boy_age", "boy_country_id", "boy_gold_num",  "boy_platform_type", "boy_type", "boy_pay_status","boy_status", "boy_eroticism_behavior", "boy_sign_eroticism", "boy_channel")
    // 女性用户信息列表名称
    val girlNames = Seq("target_user_id", "girl_app_id", "girl_gender", "girl_age", "girl_country_id", "girl_gold_num",  "girl_platform_type", "girl_type", "girl_pay_status","girl_status", "girl_eroticism_behavior", "girl_sign_eroticism", "girl_channel")

    // 近期内的登录用户（近期活跃用户）
    val aliveUserDF = sql(
      s"""SELECT user_id
         |FROM rc_live_chat_statistics.rc_user_record where dt>='${offsetDay}'""".stripMargin)
    // 统计每个近期活跃用户的登陆数量 -- user_login_count
    val loginCountDF = aliveUserDF.groupBy("user_id").agg(count("user_id").as("user_login_count"))

    // 用户的国家
    val userCountryIdDF = userDF.select("user_id", "country_id")

    // 近期视频截图记录
    val snapShotsDF = sql(
      s"""SELECT user_id,
         |location,
         |violations_rate,
         |violations_review,
         |user_eroticism,
         |violations_label=0,
         |video_type,
         |page,
         |old_new_user,
         |model,
         |partner_ship,
         |remote_gender
         |FROM rc_live_chat_statistics.rc_video_snapshots_all where dt>='${offsetDay}'
          """.stripMargin)

    // 视频记录表
    val videoRecordDF = sql(
      s"""SELECT user_id,
         |       matched_id,
         |       video_time / 1000 video_time,
         |       gender_condition,
         |       request_type,
         |       goddess_location,
         |       goddess_video,
         |       gender,
         |       dt
         |FROM rc_live_chat_statistics.rc_video_record
         |where dt >= '${offsetDay}'""".stripMargin)

    // 女神用户
    val goddessUserIdDF = sql("SELECT user_id FROM rc_video_chat.rc_goddess")
    // 普通合作女用户
    val normalUserIdDF = sql("SELECT user_id FROM rc_video_chat.rc_temp_user")

    // 用户行为埋点表
    val userRequestLocationRecordDF = sql(
      s"""select user_id,target_user_id,event_id,dt
         |from data_plat.rc_user_request_location_record
         |where dt >= '${offsetDay}'""".stripMargin)

    // 发起匹配表
    val matchRequestDF = sql(
      s"""SELECT  user_id,match_gender,create_date
         |FROM rc_live_chat_statistics.rc_match_request
         |where dt >= '${offsetDay}'""".stripMargin).distinct()

    // 发起匹配中首次发起请求的数量
    val matchRequestCountDF = matchRequestDF.groupBy("user_id")
      .agg(count("match_gender").as("boy_match_request_count"))
    //first("match_gender").as("boy_first_match_request")
    //).select("user_id", "0_boy_match_request_count", "1_boy_match_request_count", "2_boy_match_request_count")

    // 匹配成功表
    val matchSuccessDF = sql(
      s"""SELECT  user_id,match_user_id target_user_id
         |FROM rc_live_chat_statistics.rc_match_stat
         |where dt >= '${offsetDay}'""".stripMargin).distinct()

    // 好友关系表
    val userFriendDF = sql(
      s"""SELECT user_id, user_friend_id
         |FROM rc_video_chat.rc_user_friend
         |where status = 1
         |  and friend_type = 2""".stripMargin)

    // 用户充值表 （支付成功）
    val userPayRecordDF = sql(
      s"""select user_id,gold_num
         |from rc_video_chat.rc_user_pay_record
         |where verify_result=1""".stripMargin)

    // 用户充值的次数、数量、和 平均数
    val userPayDF = userPayRecordDF.groupBy("user_id").agg(
      count("gold_num").as("boy_pay_count"),
      sum("gold_num").as("boy_pay_total_get_coins"),
      avg("gold_num").as("boy_pay_avg_get_coins")
    )

    // 加金币记录表
    val addGoldRecordDF = sql(
      s"""select userId user_id,add_num,type
         |from rc_live_chat_statistics.rc_add_gold_record""".stripMargin)

    // 男性用户加金币的次数
    val addGoldDF = addGoldRecordDF.groupBy("user_id")
      .agg(count("type").as("boy_add_gold_count")
      )

    // 余额流水变动表
    val userBalanceChangeRecordDF = sql(
      s"""select  user_id,item_type,change_type,after_balance-before_balance change_gold
         |from rc_live_chat_statistics.rc_user_balance_change_record""".stripMargin)

    // 用户在不同场景下发生增加余额变动的次数和金额
    val userBalanceChangeAddDF = userBalanceChangeRecordDF.filter($"change_type" === 1)
      .groupBy("user_id")
      //.pivot("item_type")
      .agg(count("change_type").as("boy_balance_change_add_count"),
        sum("change_gold").as("boy_balance_change_add_total_gold"),
        avg("change_gold").as("boy_balance_change_add_avg_gold")
      )

    // 用户在不同场景下发生减少余额变动的次数和金额
    val userBalanceChangeSubDF = userBalanceChangeRecordDF.filter($"change_type" === 0)
      .groupBy("user_id")
      //.pivot("item_type")
      .agg(count("change_type").as("boy_balance_change_sub_count"),
        sum("change_gold").as("boy_balance_change_sub_total_gold"),
        avg("change_gold").as("boy_balance_change_sub_avg_gold")
      )

    // 男性用户在自己的好友列表中给女神送出的金币数
    val sendFriendVideoGoddessDF = sql(
      s"""select user_id,remote_user_id target_user_id,gold_num send_girl_call_gold,dt
         |from rc_live_chat_statistics.rc_goddess_goldnum_statistics
         |where dt >= '${offsetDay}' and gold_num>0 and call_mode=3""".stripMargin)

    // 男性用户在自己的好友列表中给普通合作女送出的金币数
    val sendFriendVideoNormalDF = sql(
      s"""select user_id,remote_user_id target_user_id,gold_num send_girl_call_gold,dt
         |from rc_live_chat_statistics.rc_minute_goldnum_record
         |where dt >= '${offsetDay}' and gold_num>0 and call_mode=3""".stripMargin)

    /***
     * 产生标签
     ***/

    // 产生标签函数（取决于用户近期视频截图中违规截图数占总截图数的比例，以及用户本身色情属性计算决定）
    val getLabelUDF = udf((violations_count:Int, sexual_count:Int, snapshots_count:Int) =>
    {
      var res:Int = -1
      val rate: Float = (violations_count * 2 + sexual_count * 1) / (snapshots_count * 2)
      if(rate>0.5){
        res = 1
      }else{
        res = 0
      }
    })

    // 产生标签
    val labelDF = videoRecordDF.join(userDF, "user_id")
      .withColumn("label", getLabelUDF($"violations_count", $"sexual_count", $"snapshots_count"))
      .filter($"label" =!= -1)
      .groupBy("user_id").agg(max("label").as("label"))


    /***
     * 特征合并部分
     ***/

    val resDF = labelDF

    // 划分features
    val vectorNames = resDF.columns.filterNot(_.contains("user_id")).filterNot(_.contains("label"))
    val vectorAssembler = new VectorAssembler()
      .setInputCols(vectorNames)
      .setOutputCol("featuresOut")

    val vectorAssemblerDF = vectorAssembler.setHandleInvalid("keep").transform(resDF)
    vectorAssemblerDF.printSchema()

    // 向量特转规范的离散特征
    val featureIndexer = new VectorIndexer()
      .setInputCol("featuresOut")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(quantileSize)
      .fit(vectorAssemblerDF)

    val data = featureIndexer.transform(vectorAssemblerDF).select("user_id", "indexedFeatures", "label")

//    // 字符串特征转离散特征
//    val stringIndexerTransformers = resDF.columns.filter(_.contains("category")).flatMap((columnName: String) => {
//      val stringIndexer = new StringIndexer()
//        .setInputCol(columnName)
//        .setOutputCol(s"${columnName}_idx")
//        .setHandleInvalid("skip")
//      Array(stringIndexer)
//    })

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
      .setFeaturesCol("indexedFeatures")

    // 讲数据进行拆分。分为训练集和测试集
    val Array(trainDF, testDF) = data.randomSplit(Array(0.8, 0.2))

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
    println(s"${binaryClassificationEvaluator.getMetricName} = $classificationMetric" +
      "\ncolumn count: " + data.head().getAs[Vector]("assembler").size +
      "\nfinal data label 0 count: " + data.filter(col("label") === 0).count() +
      "\nfinal data label 1 count: " + data.filter(col("label") === 1).count()
    )

    /***
     * 模型保存
     ***/

//    deleteFile(spark, lr_model_savepath, today, historyOffsetDay)
    model.save(lr_model_savepath + today)
    println(s"Model has been saved to ${lr_model_savepath + today} !")

  }
}
