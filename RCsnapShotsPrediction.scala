package com.rc.platform.task

import com.rc.platform.config.SpeckConfig
import com.rc.platform.util.{AlertUtils, RedisDao}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.recommendation.{ALS, ALSModel}
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.sql.Row
import org.apache.spark.sql.functions.{udf, _}

import scala.collection.mutable.ArrayBuffer
import scala.math.{pow, sqrt}

object RCsnapShotsPrediction extends TaskSchedule {
  //  val sendGiftScore = 2.0
  //  val sendFriendVideoCallScore = 1.0
  //  val sendGoddessWallCallScore = 1.0
  val matchVideoScore = 1.0
  val sendTextChatScore = 0.1
  val sendEffectScore = 0.2
  val sendVoiceScore = 0.2
  val addFriendScore = 0.4
  val likeScore = 0.2
  val clickAvatorScore = 0.2
  val collectScore = 0.5
  val onlineRemindScore = 0.5

  val alpha = 1.0
  val similarProductsSize = 100
  val redis = new RedisDao

  def pickTopN(n: Int, iterable: Iterable[(Int, Float)]): Seq[Int] = {
    val seq = iterable.toSeq
    implicit val ord: Ordering[(Int, Float)] = Ordering.by(_._2)
    val q = collection.mutable.PriorityQueue[(Int, Float)](seq.take(n): _*)(ord.reverse)
    seq.drop(n).foreach(v => {
      q += v
      q.dequeue()
    })
    q.dequeueAll.reverse.map(f => f._1)
  }

  def cosineSimilarity(vec1: Array[Float], vec2: Seq[Float]): Float = {
    val size = vec1.length
    var molecular = 0.0
    var demolecularLeft = 0.0
    var demolecularRight = 0.0
    for (i <- 0 until size) {
      molecular += vec1(i) * vec2(i)
      demolecularLeft += pow(vec1(i), 2)
      demolecularRight += pow(vec2(i), 2)
    }
    val result = molecular / (sqrt(demolecularLeft) * sqrt(demolecularRight))
    result.toFloat
  }

  def main(args: Array[String]): Unit = {

    val spark = SpeckConfig.getSpeckConfigInstance(this.getClass.getSimpleName.filter(!_.equals('$')))
    import spark.implicits._
    import spark.sql

    val offset = args(0).toInt
    val offsetDay = getOffsetDay(offset)
    val today = getToday()
    val historyOffset = -180
    val historyOffsetDay = getOffsetDay(historyOffset)

    val goddessUserIdDF = sql("SELECT user_id target_user_id FROM  rc_video_chat.rc_goddess")
    val normalUserIdDF = sql("SELECT user_id target_user_id FROM  rc_video_chat.rc_temp_user")
    val totalGirlsDF = goddessUserIdDF.union(normalUserIdDF)

    val userCountryIdDF = sql(
      s"""SELECT  user_id target_user_id, country_id
         |FROM rc_video_chat.rc_user""".stripMargin)

    val girlCountryIdMap = totalGirlsDF.join(userCountryIdDF, "target_user_id").as[(Int, Int)].collect().toMap
    val bcGirlCountryIdMap = spark.sparkContext.broadcast(girlCountryIdMap)

    val userRequestLocationRecordDF = sql(
      s"""select user_id,target_user_id,dt
         |from data_plat.rc_user_request_location_record
         |where dt >= '${offsetDay}'""".stripMargin)
    //avg sendGiftRatingDF 330 gold ,91657 total
    val sendGiftRatingDF = sql(
      s"""select user_id target_user_id, send_user_id user_id,gift_gold consume_gold,gift_gold/10 rating,dt
         |from rc_live_chat_statistics.rc_new_user_gift_detail
         |where dt >= '${offsetDay}'""".stripMargin)
      .select("user_id", "target_user_id", "rating")
    //    sendGiftRealDF.groupBy("dt").agg(avg("consume_gold")).show
    //avg sendFriendVideoCallRatingDF 114 gold,922159 total
    val sendFriendVideoGoddessRatingDF = sql(
      s"""select user_id,remote_user_id target_user_id,gold_num consume_gold,gold_num/10 rating,dt
         |from rc_live_chat_statistics.rc_goddess_goldnum_statistics
         |where dt >= '${offsetDay}' and gold_num>0 and call_mode=3""".stripMargin)
    val sendFriendVideoNormalRatingDF = sql(
      s"""select user_id,remote_user_id target_user_id,gold_num consume_gold,gold_num/10 rating,dt
         |from rc_live_chat_statistics.rc_minute_goldnum_record
         |where dt >= '${offsetDay}' and gold_num>0 and call_mode=3""".stripMargin)
    val sendFriendVideoCallRatingDF = sendFriendVideoGoddessRatingDF.union(sendFriendVideoNormalRatingDF)
      .select("user_id", "target_user_id", "rating")

    //    sendFriendVideoCallRatingDF.groupBy("dt").agg(avg("consume_gold")).show
    //avg sendGoddessWallCallRatingDF 60 gold ,213716 total
    val sendGoddessWallCallRatingDF = sql(
      s"""select user_id,remote_user_id target_user_id,gold_num consume_gold,gold_num/10 rating,dt
         |from rc_live_chat_statistics.rc_goddess_goldnum_statistics
         |where dt >= '${offsetDay}' and gold_num>0 and call_mode=1""".stripMargin)
      .select("user_id", "target_user_id", "rating")

    //    sendGoddessWallCallRatingDF.groupBy("dt").agg(avg("consume_gold")).show
    //avg matchVideoRecordDF 16s ,3991067 total, 9 or 0 coins
    val matchVideoRecordDF = sql(
      s"""SELECT user_id,
         |       matched_id        target_user_id,
         |       video_time / 1000 video_time,
         |       dt
         |FROM rc_live_chat_statistics.rc_video_record
         |where dt >= '${offsetDay}'
         |  and request_type = 0
         |  and goddess_location = 2
         |  and goddess_video = 1""".stripMargin).filter($"video_time" >= 10)
      .withColumn("rating", lit(matchVideoScore))
      .select("user_id", "target_user_id", "rating")

    //    matchVideoRecordDF.groupBy("dt").agg(avg("video_time")).show
    //  sendTextChatRatingDF  4093593 total
    val sendTextChatRatingDF = userRequestLocationRecordDF.filter($"event_id" === "7-4-6-7" || $"event_id" === "5-1-1-4")
      .withColumn("rating", lit(sendTextChatScore))
      .select("user_id", "target_user_id", "rating")
    //  sendEffectRatingDF  34684 total
    val sendEffectRatingDF = userRequestLocationRecordDF.filter($"event_id" === "5-1-1-2")
      .withColumn("rating", lit(sendEffectScore))
      .select("user_id", "target_user_id", "rating")
    //  sendVoiceRatingDF  1625 total
    val sendVoiceRatingDF = userRequestLocationRecordDF.filter($"event_id" === "5-1-1-7")
      .withColumn("rating", lit(sendVoiceScore))
      .select("user_id", "target_user_id", "rating")
    //  addFriendRatingDF  459351 total
    val addFriendRatingDF = userRequestLocationRecordDF.filter($"event_id" === "5-1-1-14")
      .withColumn("rating", lit(addFriendScore))
      .select("user_id", "target_user_id", "rating")
    //  likeRatingDF  242243 total
    val likeRatingDF = userRequestLocationRecordDF.filter($"event_id" === "7-9-12-3" || $"event_id" === "5-1-1-11")
      .withColumn("rating", lit(likeScore))
      .select("user_id", "target_user_id", "rating")
    //  clickAvatorRatingDF  242243 total
    val clickAvatorRatingDF = userRequestLocationRecordDF.filter($"event_id" === "7-1-1-1")
      .withColumn("rating", lit(clickAvatorScore))
      .select("user_id", "target_user_id", "rating")
    //  collectRatingDF  23369 total
    val collectRatingDF = userRequestLocationRecordDF.filter($"event_id" === "7-1-1-2" || $"event_id" === "7-4-6-12")
      .withColumn("rating", lit(collectScore))
      .select("user_id", "target_user_id", "rating")
    //  onlineRemindRatingDF  33606 total
    val onlineRemindRatingDF = userRequestLocationRecordDF.filter($"event_id" === "7-4-6-13")
      .withColumn("rating", lit(onlineRemindScore))
      .select("user_id", "target_user_id", "rating")

    val sumData = sendGiftRatingDF
      .union(sendFriendVideoCallRatingDF)
      .union(sendGoddessWallCallRatingDF)
      .union(matchVideoRecordDF)
      .union(sendTextChatRatingDF)
      .union(sendEffectRatingDF)
      .union(sendVoiceRatingDF)
      .union(addFriendRatingDF)
      .union(likeRatingDF)
      .union(clickAvatorRatingDF)
      .union(collectRatingDF)
      .union(onlineRemindRatingDF)
      .join(totalGirlsDF, "target_user_id")
      .groupBy("user_id", "target_user_id").agg(sum($"rating").as("sum"))
      .select($"user_id", $"target_user_id", $"sum")
    val filterLargeUDF = udf((rating: Double) => {
      if (rating < 100.0) rating else 100.0
    })
    val ratings = sumData.withColumn("sumrating", filterLargeUDF(sumData("sum")))

    val boyLikeSumRatingDF = ratings
      .join(userCountryIdDF, Seq("target_user_id"))
      .groupBy("user_id", "country_id")
      .agg(sum("sumrating"))

    def sortByUdf = udf((structs: Seq[Row]) => {
      val sortedStruct = structs.sortBy(str => str.getAs[Double]("sum(sumrating)"))(Ordering[Double].reverse)
      sortedStruct.map(str => str.getAs[Int]("country_id")).take(3).mkString(",")
    })

    val countrylikeDF = boyLikeSumRatingDF
      .select(col("user_id"), struct(col("country_id"), col("sum(sumrating)")).as("struct"))
      .groupBy("user_id")
      .agg(sortByUdf(collect_list("struct")).as("country_list"))

    val aliveUserDF = sql(
      s"""SELECT user_id
         |FROM rc_live_chat_statistics.rc_user_record where dt >='${offsetDay}'""".stripMargin)
    val girlAliveUserDF = aliveUserDF.filter($"gender" === 2).withColumnRenamed("user_id", "target_user_id")

    val useraliveCountryIdDF = userCountryIdDF
      .join(girlAliveUserDF, "target_user_id")
      .join(goddessUserIdDF, "target_user_id")
      .distinct()

    val countryIdGoddessIdMap = useraliveCountryIdDF
      .groupBy("country_id").agg(collect_list("target_user_id").as("goddess_list"))
      .select("country_id", "goddess_list").as[(String, Array[Int])].collect().toMap

    val generateAreaResUDF = udf((country_list: String) => {
      val goddessList = ArrayBuffer[Int]()
      country_list.split(",").foreach {
        countrylike_list =>
          if (countryIdGoddessIdMap.keySet.contains(countrylike_list)) {
            goddessList ++= countryIdGoddessIdMap(countrylike_list)
          }
      }
      goddessList
    })
    val generateCountryListUDF = udf((user_id: Int, country_list: String) => {
      redis.set("rc_recommend_likecountry_" + user_id, country_list, 3600 * 24 * 7)
      "rc_recommend_likecountry_" + user_id + ":" + country_list
    })
    val userIdGoddessIdDF = countrylikeDF
      .withColumn("goddesslist", generateAreaResUDF($"country_list"))
      .select("user_id", "goddesslist", "country_list")
      .withColumn("res", generateCountryListUDF($"user_id", $"country_list"))

    deleteFile(spark, "s3://bigdata-rc/algorithm-data/recommend/ALS/userIdGoddessId/", today, historyOffsetDay)
    userIdGoddessIdDF.write.save("s3://bigdata-rc/algorithm-data/recommend/ALS/userIdGoddessId/" + today)


    /**
     * ALS codes starts here
     * ALS (alternating least square is a popular model that spark-ml use for 'Collaborative filtering'
     */

    val ranks = Array(10) //numbers of latent factor used to predict missing entries of user-item matrics, the default is 10
    val iters = Array(15) //the default is 10
    val regParams = Array(0.01)

    val als = new ALS().setImplicitPrefs(true).setUserCol("user_id").setItemCol("target_user_id").setRatingCol("sumrating").setAlpha(alpha)

    val paramGrids = new ParamGridBuilder()
      .addGrid(als.rank, ranks)
      .addGrid(als.maxIter, iters)
      .addGrid(als.regParam, regParams)
      .build() //build return Array[ParamMap]

    var bestModel: Option[ALSModel] = None
    var bestParam: Option[ParamMap] = None

    for (paramMap <- paramGrids) {
      val model = als.fit(ratings, paramMap)
      bestModel = Some(model)
      bestParam = Some(paramMap)
    }
    val arraySumUDF = udf((array: Seq[Float]) => {
      array.sum
    })
    //    val K = 1000
    val userFactors = bestModel.get.userFactors.withColumnRenamed("id", "user_id")
    val productFeatures = bestModel.get.itemFactors
      .withColumnRenamed("id", "target_user_id")
      .withColumn("featuresSum", arraySumUDF($"features"))
      .filter($"featuresSum" =!= 0.0)
      .drop("featuresSum")

    val productFeaturesMap = productFeatures.select($"target_user_id", $"features").as[(Int, Array[Float])].collect.toMap
    val findSimilarProductsUDF = udf((target_user_id: Int, features: Seq[Float]) => {
      val sims = productFeaturesMap.map { case (id, factor) =>
        val factorVector = factor
        var similarity = -1.0f
        if (bcGirlCountryIdMap.value.getOrElse(target_user_id, -1) == bcGirlCountryIdMap.value.getOrElse(id, -1)) {
          similarity = cosineSimilarity(factorVector, features)
        }
        (id, similarity)
      }
      val simResult: Seq[Int] = pickTopN(similarProductsSize, sims)
      redis.set("rc_recommend_hotvideos_similar_girls_" + target_user_id, simResult.mkString(","), 3600 * 24 * 7)
      simResult
    })
    val similarProducts = productFeatures
      .withColumn("similarProducts", findSimilarProductsUDF(productFeatures("target_user_id"), productFeatures("features")))
      .select($"target_user_id", $"similarProducts")
    val similarResultMap = similarProducts.as[(Int, Array[Int])].collect.toMap
    val bcSimilarResultMap = spark.sparkContext.broadcast(similarResultMap)
    val usrAvgRatingListDF = ratings.groupBy("user_id").agg(mean("sumrating").as("avgRating"))
    val avgRatingJoinRatingDF = usrAvgRatingListDF.join(ratings, "user_id")
    val filterBigThanAvgRatingDF = avgRatingJoinRatingDF.join(productFeatures, "target_user_id").filter($"sumrating" >= $"avgRating")
    val sortAndAggUdf = udf((structs: Seq[Row]) => {
      val sortedStruct = structs.sortBy(str => str.getAs[Double]("sumrating"))(Ordering[Double].reverse)
      sortedStruct.map(str => str.getAs[Int]("target_user_id"))
    })
    val userVideos = filterBigThanAvgRatingDF.select(col("user_id"), struct(col("target_user_id"), col("sumrating")).as("struct"))
      .groupBy("user_id").agg(sortAndAggUdf(collect_list("struct")).as("vids"))

    val generateSimilarProductsUDF = udf((uuid: String, vids: Seq[Int]) => {
      var res = ""
      import scala.collection.mutable.LinkedHashSet
      //      val uuidSawVidKey = "uuid.played.new." + uuid
      //      val uuidSawVidValue = getRedisValue(sparkRedis, uuidSawVidKey).split(",").toSet
      var set: LinkedHashSet[Int] = LinkedHashSet()
      for (i <- 1 until similarProductsSize; product <- vids if set.size < similarProductsSize) {
        val similarResult = bcSimilarResultMap.value.get(product).get
        if (i < similarResult.size) {
          set += similarResult(i)
        }
      }
      //      val simResultAvgTime = set.take(similarProductsSize).map { vid => (vid, vidAvgTimeMap(vid.toString)) }
      //      val result = simResultAvgTime.toList.sortWith(_._2 > _._2).map(_._1) ++ set.takeRight(similarProductsSize - similarProductsSize)
      val list = set.take(similarProductsSize).map(_.toString).toList
      redis.del("rc_recommend_hotvideos_userid_" + uuid)
      redis.rpush("rc_recommend_hotvideos_userid_" + uuid, list)
      redis.expire("rc_recommend_hotvideos_userid_" + uuid, 3600 * 24 * 7)
      res = "rc_recommend_hotvideos_userid_" + uuid + ":" + list.mkString(",")
      res
    })
    val recommendedProductsPerUser = userVideos.withColumn("res", generateSimilarProductsUDF($"user_id", $"vids"))
    deleteFile(spark, "s3://bigdata-rc/algorithm-data/recommend/ALS/recommendRes/", today, historyOffsetDay)
    recommendedProductsPerUser.select("user_id", "res").write.save("s3://bigdata-rc/algorithm-data/recommend/ALS/recommendRes/" + today)
    //parquet save productFeatures
    deleteFile(spark, "s3://bigdata-rc/algorithm-data/recommend/ALS/productFeatures/", today, historyOffsetDay)
    productFeatures.write.save("s3://bigdata-rc/algorithm-data/recommend/ALS/productFeatures/" + today)

    //parquet save userFeatures
    deleteFile(spark, "s3://bigdata-rc/algorithm-data/recommend/ALS/userFeatures/", today, historyOffsetDay)
    userFactors.write.save("s3://bigdata-rc/algorithm-data/recommend/ALS/userFeatures/" + today)

    val dfsPath = "s3://bigdata-rc/algorithm-data/recommend/ALS/recommendRes/" + today
    AlertUtils.checkRedis(spark, dfsPath, "recommend/ALS/recommendRes")


  }
}
