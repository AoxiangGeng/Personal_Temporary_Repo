
package com.rc.platform.task

import com.rc.platform.config.SpeckConfig
import com.rc.platform.config.EventMapConfig.eventMap
import com.rc.platform.task.RCAlsTrain.deleteFile
import com.rc.platform.util.RedisDao
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.ml.feature.{OneHotEncoder, OneHotEncoderEstimator, QuantileDiscretizer, StringIndexer}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{udf, _}
import org.apache.spark.ml.feature.VectorAssembler
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
    val offsetDay = "2019-12-08"
    // 获取今日时间
    val today = getToday()
    // 分桶数
    val quantileSize = 15
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

    // 视频截图记录, even表
    val snapShotsEven = sql(
      s"""SELECT user_id,
         |gender,
         |app_id,
         |violations_review,
         |violation_rate,
         |gender_rate,
         |old_new_user,
         |partner_ship,
         |room_id,
         |location,
         |user_eroticism,
         | violations_label=0
         |FROM rc_live_chat_statistics.rc_video_snapshots_even
          """.stripMargin)

    // 视频截图记录, odd表
    val snapShotsOdd = sql(
      s"""SELECT user_id,
         |gender,
         |app_id,
         |violations_review,
         |violation_rate,
         |gender_rate,
         |old_new_user,
         |partner_ship,
         |room_id,
         |location,
         |user_eroticism,
         | violations_label=0
         |FROM rc_live_chat_statistics.rc_video_snapshots_odd
          """.stripMargin)
    // 合并截图记录表
    val snapShotsDF = snapShotsEven.union(snapShotsOdd)
    // 好友关系表
    val userFriendDF = sql(
      s"""SELECT user_id, user_friend_id
         |FROM rc_video_chat.rc_user_friend
         |where status = 1
         |  and friend_type = 2""".stripMargin)

    // 女神用户
    val goddessUserIdDF = sql("SELECT user_id target_user_id FROM  rc_video_chat.rc_goddess")
    // 普通合作女用户
    val normalUserIdDF = sql("SELECT user_id target_user_id FROM  rc_video_chat.rc_temp_user")
    // 用户的国家
    val userCountryIdDF = userDF.withColumnRenamed("user_id", "target_user_id")
      .select("target_user_id", "country_id")


    /***
     * 特征合并部分
     ***/






    /***
     * 模型训练
     ***/

    // 构建模型框架
    val data = sc.textFile("RCPornSnapshots.data").map {//是否将数据保存到本地进行训练（有待商榷）
      line =>
        val parts = line.split(",")
        val y = parts(0)
        val xs = parts(1)
        LabeledPoint(parts(0).toDouble,Vectors.dense(parts(1).split(" ").map { _.toDouble}))
    }.cache()

    // 讲数据进行拆分。分为训练集和测试集(具体操作有待商榷)
    val train2test = data.randomSplit(Array(0.8,0.2), 1)
    val lrs = new LogisticRegression()
    lrs.setIntercept(true)//设置true-有截距
    lrs.optimizer.setNumIterations(100)//设置迭代次数
    lrs.setRegParam(0.3) //正则化项系数
    lrs.setElasticNetParam(0.0) //L2范数
    lrs.setStandardization(true) //将feature标准化
    lrs.setLabelCol("label")
    lrs.setFeaturesCol("assembler")

    val model = lrs.fit(train2test(0))//训练模型
    print("截距："+model.intercept)
    print("weights:"+model.weights)

    //对样本进行测试
    val prediction = model.transform(train2test(1).map {_.features })//等到预测值
    val predictionAndLabel = prediction.zip(train2test(1).map { _.label})//（预测值，真实值）
    val loss = predictionAndLabel.map({
      case (p,v)=>
        val error = p-v
        Math.abs(error)
    }).reduce(_+_)
    val error = loss/train2test(1).count()
    println("平均误差："+error)

    //        model.save(sc, "./mymodel")
    //        val mymodel = LinearRegressionModel.load(sc, "./mymodel") //读取模型，加载模型

    val sqlContext = new SQLContext(sc)
    sqlContext.read.parquet("./myModel/data").show
    sc.stop


    /***
     * 模型持久化
     ***/



  }
}
