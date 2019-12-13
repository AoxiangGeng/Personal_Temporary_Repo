
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

    //男性用户信息列表名称
    val boyNames = Seq("user_id", "boy_app_id", "boy_gender", "boy_age", "boy_country_id", "boy_gold_num",  "boy_platform_type", "boy_type", "boy_pay_status","boy_status", "boy_eroticism_behavior", "boy_sign_eroticism", "boy_channel")
    //女性用户信息列表名称
    val girlNames = Seq("target_user_id", "girl_app_id", "girl_gender", "girl_age", "girl_country_id", "girl_gold_num",  "girl_platform_type", "girl_type", "girl_pay_status","girl_status", "girl_eroticism_behavior", "girl_sign_eroticism", "girl_channel")
    //7天内的登录用户
    val aliveUserDF = sql(
      s"""SELECT user_id
         |FROM rc_live_chat_statistics.rc_user_record where dt>='${offsetDay}'""".stripMargin)
    // 登录用户的数量
    val loginCountDF = aliveUserDF.groupBy("user_id").agg(count("user_id").as("boy_login_count"))
    // 女神用户
    val goddessUserIdDF = sql("SELECT user_id target_user_id FROM  rc_video_chat.rc_goddess")
    // 普通合作女用户
    val normalUserIdDF = sql("SELECT user_id target_user_id FROM  rc_video_chat.rc_temp_user")
    // 7天内登录的女性用户 去重
    val totalGirlsDF = goddessUserIdDF.union(normalUserIdDF)
      .join(aliveUserDF.withColumnRenamed("user_id", "target_user_id"), Seq("target_user_id")).distinct()
    // 用户的国家
    val userCountryIdDF = userDF.withColumnRenamed("user_id", "target_user_id")
      .select("target_user_id", "country_id")
    // 将女孩的国家转成广播变量
    val girlCountryIdMap = totalGirlsDF.join(userCountryIdDF, "target_user_id").as[(Int, Int)].collect().toMap
    val bcGirlCountryIdMap = spark.sparkContext.broadcast(girlCountryIdMap)
    // 视频记录表
    val videoRecordDF = sql(
      s"""SELECT user_id,
         |       matched_id        target_user_id,
         |       video_time / 1000 video_time,
         |       gender_condition,
         |       request_type,
         |       goddess_location,
         |       goddess_video,
         |       gender,
         |       dt
         |FROM rc_live_chat_statistics.rc_video_record
         |where dt >= '${offsetDay}'""".stripMargin)
    // 用户行为埋点表
    val userRequestLocationRecordDF = sql(
      s"""select user_id,target_user_id,event_id,dt
         |from data_plat.rc_user_request_location_record
         |where dt >= '${offsetDay}'""".stripMargin)
    //得到女孩国家的ID
    val getGirlCountryIdUDF = udf((girlUserId: Int) => {
      bcGirlCountryIdMap.value.getOrElse(girlUserId, 0)
    })
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
    // 用户充值表
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
    //加金币记录表
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
    // 男性对不同国家女性产生的情况
    def generateBoyPivotDF(inputDF: DataFrame, usefulColumn: String, countName: String, countDistinctName: String, sumName: String, avgName: String) = {
      inputDF.join(totalGirlsDF, "target_user_id")
        //.withColumn("girl_country_id", getGirlCountryIdUDF($"target_user_id"))
        //.filter($"girl_country_id" =!= 0)
        .groupBy("user_id")
        //.pivot("girl_country_id")
        .agg(
          count("target_user_id").as(countName),
          countDistinct("target_user_id").as(countDistinctName),
          sum(usefulColumn).as(sumName),
          avg(usefulColumn).as(avgName)
        )
    }
    // 不同国家女性的的情况
    def generateGirlPivotDF(inputDF: DataFrame, usefulColumn: String, countName: String, countDistinctName: String, sumName: String, avgName: String) = {
      inputDF.join(totalGirlsDF, "target_user_id")
        .join(userDF, "user_id")
        //.filter($"country_id" =!= 0)
        .groupBy("target_user_id")
        //.pivot("country_id")
        .agg(
          count("user_id").as(countName),
          countDistinct("user_id").as(countDistinctName),
          sum(usefulColumn).as(sumName),
          avg(usefulColumn).as(avgName)
        )
    }
    // 送礼记录表 男生给女生赠礼的金额
    val sendGiftDF = sql(
      s"""select user_id target_user_id, send_user_id user_id,gift_gold,dt
         |from rc_live_chat_statistics.rc_new_user_gift_detail
         |where dt >= '${offsetDay}'""".stripMargin)
    // 男生对不同国家女生送礼物的总次数，总金额 平均金额
    val sendGiftPivotDF = generateBoyPivotDF(sendGiftDF,
      "gift_gold",
      "boy_send_gift_count",
      "boy_send_gift_distinct_count",
      "boy_send_gift_total_gold",
      "boy_send_gift_avg_gold")
    // 男性用户在自己的好友列表中给女神送出的金币数
    val sendFriendVideoGoddessDF = sql(
      s"""select user_id,remote_user_id target_user_id,gold_num send_girl_call_gold,dt
         |from rc_live_chat_statistics.rc_goddess_goldnum_statistics
         |where dt >= '${offsetDay}' and gold_num>0 and call_mode=3""".stripMargin)
    //男性用户在自己的好友列表中给普通合作女送出的金币数
    val sendFriendVideoNormalDF = sql(
      s"""select user_id,remote_user_id target_user_id,gold_num send_girl_call_gold,dt
         |from rc_live_chat_statistics.rc_minute_goldnum_record
         |where dt >= '${offsetDay}' and gold_num>0 and call_mode=3""".stripMargin)
    // 将女神与合作女合并
    val sendFriendVideoCallDF = sendFriendVideoGoddessDF.union(sendFriendVideoNormalDF)
    // 男性用户给好友女生的送金币的次数 总计金币数和平均金币数
    val sendFriendVideoCallPivotDF = generateBoyPivotDF(sendFriendVideoCallDF,
      "send_girl_call_gold",
      "boy_send_call_minute_count",
      "boy_send_call_distinct_minute_count",
      "boy_send_call_minute_total_gold",
      "boy_send_call_minute_avg_gold")
    // 男性用户主动拨打女神用户按分钟计费情况
    val sendFriendVideoRecordDF = videoRecordDF.filter($"request_type" === 1 && $"goddess_video" === 2)
    // 男性 用户按视频时长 给女神 续播的次数 总时长 平均时长
    val sendFriendVideoRecordPivotDF = generateBoyPivotDF(
      sendFriendVideoRecordDF,
      "video_time",
      "boy_send_call_record_count",
      "boy_send_call_record_distinct_count",
      "boy_send_call_record_total_time",
      "boy_send_call_record_avg_time")
    // 男性用户拨打女神墙消耗的金币数
    val sendGoddessWallCallDF = sql(
      s"""select user_id,remote_user_id target_user_id,gold_num send_goddess_wall_call_gold,dt
         |from rc_live_chat_statistics.rc_goddess_goldnum_statistics
         |where dt >= '${offsetDay}' and gold_num>0 and call_mode=1""".stripMargin)
    // 男性用户拨打女神墙的次数 总时长 平均时长
    val sendGoddessWallCallPivotDF = generateBoyPivotDF(
      sendGoddessWallCallDF,
      "send_goddess_wall_call_gold",
      "boy_send_goddess_wall_call_minute_count",
      "boy_send_goddess_wall_call_distinct_minute_count",
      "boy_send_goddess_wall_call_total_minute_time",
      "boy_send_goddess_wall_call_avg_minute_time")
    // 男性用户匹配女神
    val matchVideoRecordDF = videoRecordDF.filter($"request_type" === 0 && $"goddess_location" === 2 && $"goddess_video" === 1)
    // 男性用户按照视频时间进行匹配的次数 匹配总时长 匹配的平均时长
    val matchVideoRecordPivotDF = generateBoyPivotDF(
      matchVideoRecordDF,
      "video_time",
      "boy_match_video_record_count",
      "boy_match_video_record_distinct_count",
      "boy_match_video_record_total_time",
      "boy_match_video_record_avg_time")
    // 男性用户根据女性国家进行透视 数量
    def generateBoyPivotCountDF(inputDF: DataFrame, countName: String, countDistinctName: String) = {
      inputDF.join(totalGirlsDF, "target_user_id")
        //.withColumn("girl_country_id", getGirlCountryIdUDF($"target_user_id"))
        //.filter($"girl_country_id" =!= 0)
        .groupBy("user_id")
        //.pivot("girl_country_id")
        .agg(
          count("target_user_id").as(countName),
          countDistinct("target_user_id").as(countDistinctName)
        )
    }
    // 女性用户根据国家进行透视数量
    def generateGirlPivotCountDF(inputDF: DataFrame, countName: String, countDistinctName: String) = {
      inputDF.join(totalGirlsDF, "target_user_id")
        .join(userDF, "user_id")
        //.filter($"country_id" =!= 0)
        .groupBy("target_user_id")
        //.pivot("country_id")
        .agg(
          count("user_id").as(countName),
          countDistinct("user_id").as(countDistinctName)
        )
    }
    // 男性用户发送文字消息
    val sendTextChatDF = userRequestLocationRecordDF.filter($"event_id" === "7-4-6-7" || $"event_id" === "5-1-1-4")
    // 男性发送文字消息的数量
    val sendTextChatPivotDF = generateBoyPivotCountDF(
      sendTextChatDF,
      "boy_send_text_count",
      "boy_send_text_distinct_count")
    // 男性用户发送特征的次数
    val sendEffectDF = userRequestLocationRecordDF.filter($"event_id" === "5-1-1-2")
    val sendEffectPivotDF = generateBoyPivotCountDF(
      sendEffectDF,
      "boy_send_effect_count",
      "boy_send_effect_distinct_count")
    // 男性用户发送语音的次数
    val sendVoiceDF = userRequestLocationRecordDF.filter($"event_id" === "5-1-1-7")
    val sendVoicePivotDF = generateBoyPivotCountDF(
      sendVoiceDF,
      "boy_send_voice_count",
      "boy_send_voice_distinct_count")
    // 男性用户加好友的次数
    val addFriendDF = userRequestLocationRecordDF.filter($"event_id" === "5-1-1-14")
    val addFriendPivotDF = generateBoyPivotCountDF(
      addFriendDF,
      "add_friend_count",
      "boy_add_friend_distinct_count")
    // 男性用户点赞的次数
    val likeDF = userRequestLocationRecordDF.filter($"event_id" === "7-9-12-3" || $"event_id" === "5-1-1-11")
    val likePivotDF = generateBoyPivotCountDF(
      likeDF,
      "boy_like_count",
      "boy_like_distinct_count")
    // 男性用户 由别的页面进入好友头像页的次数
    val clickAvatorDF = userRequestLocationRecordDF.filter($"event_id" === "7-1-1-1")
    val clickAvatorPivotDF = generateBoyPivotCountDF(
      clickAvatorDF,
      "boy_click_avator_count",
      "boy_click_avator_distinct_count")
    // 男性用户点击收藏 点击好友列表置顶的次数
    val collectDF = userRequestLocationRecordDF.filter($"event_id" === "7-1-1-2" || $"event_id" === "7-4-6-12")
    val collectPivotDF = generateBoyPivotCountDF(
      collectDF,
      "boy_collect_count",
      "boy_collect_distinct_count")
    // 男性用户点击上线提醒次数
    val onlineRemindDF = userRequestLocationRecordDF.filter($"event_id" === "7-4-6-13")
    val onlineRemindPivotDF = generateBoyPivotCountDF(
      onlineRemindDF,
      "boy_online_remind_count",
      "boy_online_remind_distinct_count")
    // 男性用户匹配成功人的次数
    val matchSuccessPivotDF = generateBoyPivotCountDF(
      matchSuccessDF,
      "boy_match_success_count",
      "boy_match_success_distinct_count")
    // 男性用户好友的数量
    val userFriendPivotDF = generateBoyPivotCountDF(
      userFriendDF.withColumnRenamed("user_friend_id", "target_user_id"),
      "boy_friend_count",
      "boy_friend_distinct_count")

    // 男性用户画像
    val boyProfileDF = aliveUserDF.distinct().join(userDF.toDF(boyNames: _*), Seq("user_id"))
      .join(loginCountDF, Seq("user_id"), "left_outer")
      .join(matchRequestCountDF, Seq("user_id"), "left_outer")
      .join(userPayDF, Seq("user_id"), "left_outer")
      .join(addGoldDF, Seq("user_id"), "left_outer")
      .join(userBalanceChangeAddDF, Seq("user_id"), "left_outer")
      .join(userBalanceChangeSubDF, Seq("user_id"), "left_outer")
      .join(sendGiftPivotDF, Seq("user_id"), "left_outer")
      .join(sendFriendVideoCallPivotDF, Seq("user_id"), "left_outer")
      .join(sendFriendVideoRecordPivotDF, Seq("user_id"), "left_outer")
      .join(sendGoddessWallCallPivotDF, Seq("user_id"), "left_outer")
      .join(matchVideoRecordPivotDF, Seq("user_id"), "left_outer")
      .join(sendTextChatPivotDF, Seq("user_id"), "left_outer")
      .join(sendEffectPivotDF, Seq("user_id"), "left_outer")
      .join(sendVoicePivotDF, Seq("user_id"), "left_outer")
      .join(addFriendPivotDF, Seq("user_id"), "left_outer")
      .join(likePivotDF, Seq("user_id"), "left_outer")
      .join(clickAvatorPivotDF, Seq("user_id"), "left_outer")
      .join(collectPivotDF, Seq("user_id"), "left_outer")
      .join(onlineRemindPivotDF, Seq("user_id"), "left_outer")
      .join(matchSuccessPivotDF, Seq("user_id"), "left_outer")
      .join(userFriendPivotDF, Seq("user_id"), "left_outer")

    // 女性接收礼物的总数，总金额，平均金额
    val receiveGiftPivotDF = generateGirlPivotDF(sendGiftDF,
      "gift_gold",
      "girl_receive_gift_count",
      "girl_receive_gift_distinct_count",
      "girl_receive_gift_total_gold",
      "girl_receive_gift_avg_gold")
    //女生接收到好友呼叫的总次数 收到的总金币数 平均金币数
    val receiveFriendVideoCallPivotDF = generateGirlPivotDF(sendFriendVideoCallDF,
      "send_girl_call_gold",
      "girl_receive_call_minute_count",
      "girl_receive_call_distinct_minute_count",
      "girl_receive_call_minute_total_gold",
      "girl_receive_call_minute_avg_gold")
    // 女生收到视频请求的总数，接收视频的总时间 平均时间
    val receiveFriendVideoRecordPivotDF = generateGirlPivotDF(
      sendFriendVideoRecordDF,
      "video_time",
      "girl_receive_call_record_count",
      "girl_receive_call_record_distinct_count",
      "girl_receive_call_record_total_time",
      "girl_receive_call_record_avg_time")
    // 女神墙中女神按分钟收到的总次数 总时间 平均时间
    val receiveGoddessWallCallPivotDF = generateGirlPivotDF(
      sendGoddessWallCallDF,
      "send_goddess_wall_call_gold",
      "girl_receive_goddess_wall_call_minute_count",
      "girl_receive_goddess_wall_call_distinct_minute_count",
      "girl_receive_goddess_wall_call_total_minute_time",
      "girl_receive_goddess_wall_call_avg_minute_time")
    // 女生接收匹配视频的总次数 总时长 平均时长
    val receiveMatchVideoRecordPivotDF = generateGirlPivotDF(
      matchVideoRecordDF,
      "video_time",
      "girl_receive_match_video_record_count",
      "girl_receive_match_video_record_distinct_count",
      "girl_receive_match_video_record_total_time",
      "girl_receive_match_video_record_avg_time")
    // 女生收到文字消息的总次数
    val receiveTextChatPivotDF = generateGirlPivotCountDF(
      sendTextChatDF,
      "girl_receive_text_count",
      "girl_receive_text_distinct_count")
    // 女生收到加载特效行为的总次数
    val receiveEffectPivotDF = generateGirlPivotCountDF(
      sendEffectDF,
      "girl_receive_effect_count",
      "girl_receive_effect_distinct_count")
    // 女生收到语音翻译行为的总次数
    val receiveVoicePivotDF = generateGirlPivotCountDF(
      sendVoiceDF,
      "girl_receive_voice_count",
      "girl_receive_voice_distinct_count")
    // 女生收到加好友行为的总次数
    val receiveAddFriendPivotDF = generateGirlPivotCountDF(
      addFriendDF,
      "girl_receive_add_friend_count",
      "girl_receive_add_friend_distinct_count")
    //女生收到喜欢点赞的数量
    val receiveLikePivotDF = generateGirlPivotCountDF(
      likeDF,
      "girl_receive_like_count",
      "girl_receive_like_distinct_count")
    // 女生收到的进入好友头像页数量
    val receiveClickAvatorPivotDF = generateGirlPivotCountDF(
      clickAvatorDF,
      "girl_receive_click_avator_count",
      "girl_receive_click_avator_distinct_count")
    // 女生收到收藏好友列表置顶次数
    val receiveCollectPivotDF = generateGirlPivotCountDF(
      collectDF,
      "girl_receive_collect_count",
      "girl_receive_collect_distinct_count")
    // 女生收到上线提醒次数
    val receiveOnlineRemindPivotDF = generateGirlPivotCountDF(
      onlineRemindDF,
      "girl_receive_online_remind_count",
      "girl_receive_online_remind_distinct_count")

    // 女生用户画像
    val girlProfileDF = totalGirlsDF
      .join(userDF.toDF(girlNames: _*), Seq("target_user_id"))
      .join(receiveGiftPivotDF, Seq("target_user_id"), "left_outer")
      .join(receiveFriendVideoCallPivotDF, Seq("target_user_id"), "left_outer")
      .join(receiveFriendVideoRecordPivotDF, Seq("target_user_id"), "left_outer")
      .join(receiveGoddessWallCallPivotDF, Seq("target_user_id"), "left_outer")
      .join(receiveMatchVideoRecordPivotDF, Seq("target_user_id"), "left_outer")
      .join(receiveTextChatPivotDF, Seq("target_user_id"), "left_outer")
      .join(receiveEffectPivotDF, Seq("target_user_id"), "left_outer")
      .join(receiveVoicePivotDF, Seq("target_user_id"), "left_outer")
      .join(receiveAddFriendPivotDF, Seq("target_user_id"), "left_outer")
      .join(receiveLikePivotDF, Seq("target_user_id"), "left_outer")
      .join(receiveClickAvatorPivotDF, Seq("target_user_id"), "left_outer")
      .join(receiveCollectPivotDF, Seq("target_user_id"), "left_outer")
      .join(receiveOnlineRemindPivotDF, Seq("target_user_id"), "left_outer")

    // 男性用户对女生的分组 总数量 平均数量、
    def generateDF(inputDF: DataFrame, usefulColumn: String, countName: String, sumName: String, avgName: String) = {
      inputDF
        .groupBy("user_id", "target_user_id")
        .agg(
          count("target_user_id").as(countName),
          sum(usefulColumn).as(sumName),
          avg(usefulColumn).as(avgName)
        )
    }
    // 男性用户对女性（普通合作女、女神）送礼物的总次数 总金币 平均金币
    val sendGiftBoyToGirlDF = generateDF(sendGiftDF,
      "gift_gold",
      "boy_to_girl_send_gift_count",
      "boy_to_girl_send_gift_total_gold",
      "boy_to_girl_send_gift_avg_gold")
    // 男性用户对女性好友拨打的总次数 话费的总金币 平均金币
    val sendFriendVideoCallBoyToGirlDF = generateDF(sendFriendVideoCallDF,
      "send_girl_call_gold",
      "boy_to_girl_send_call_minute_count",
      "boy_to_girl_send_call_minute_total_gold",
      "boy_to_girl_send_call_minute_avg_gold")
    val sendFriendVideoRecordBoyToGirlDF = generateDF(
      sendFriendVideoRecordDF,
      "video_time",
      "boy_to_girl_send_call_record_count",
      "boy_to_girl_send_call_record_total_time",
      "boy_to_girl_send_call_record_avg_time")
    // 男性用户拨打女神墙的总次数 总时长 平均时长
    val sendGoddessWallCallBoyToGirlDF = generateDF(
      sendGoddessWallCallDF,
      "send_goddess_wall_call_gold",
      "boy_to_girl_send_goddess_wall_call_minute_count",
      "boy_to_girl_send_goddess_wall_call_total_minute_time",
      "boy_to_girl_send_goddess_wall_call_avg_minute_time")
    //男性用户匹配记录中匹配视频的总次数 总时长 平均时长
    val matchVideoRecordBoyToGirlDF = generateDF(
      matchVideoRecordDF,
      "video_time",
      "boy_to_girl_match_video_record_count",
      "boy_to_girl_match_video_record_total_time",
      "boy_to_girl_match_video_record_avg_time")
    // 按男性分组聚合数量
    def generateCountDF(inputDF: DataFrame, countName: String) = {
      inputDF
        .groupBy("user_id", "target_user_id")
        .agg(
          count("target_user_id").as(countName)
        )
    }
    // 男性给女生发送文字消息的数量
    val sendTextChatBoyToGirlDF = generateCountDF(
      sendTextChatDF,
      "boy_to_girl_send_text_count")
    // 男性给女性开启特效的次数
    val sendEffectBoyToGirlDF = generateCountDF(
      sendEffectDF,
      "boy_to_girl_send_effect_count")
    // 男性用户给女性发送语音消息的次数
    val sendVoiceBoyToGirlDF = generateCountDF(
      sendVoiceDF,
      "boy_to_girl_send_voice_count")
    //男性用户添加女性用户为好友的次数
    val addFriendBoyToGirlDF = generateCountDF(
      addFriendDF,
      "boy_to_girl_add_friend_count")
    // 男孩对女孩点赞的次数
    val likeBoyToGirlDF = generateCountDF(
      likeDF,
      "boy_to_girl_like_count")
    // 男性用户点击进入好友头像页的次数
    val clickAvatorBoyToGirlDF = generateCountDF(
      clickAvatorDF,
      "boy_to_girl_click_avator_count")
    // 男性用户点击收藏的次数
    val collectBoyToGirlDF = generateCountDF(
      collectDF,
      "boy_to_girl_collect_count")
    // 男性用户点击女性好友上线提醒业务
    val onlineRemindBoyToGirlDF = generateCountDF(
      onlineRemindDF,
      "boy_to_girl_online_remind_count")

    /***
     * 产生标签
     ***/

    // 近期视频截图记录
    val snapShotsDF = sql(
      s"""SELECT user_id,
         |location,
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

    // 分别统计截图中色情/性感/正常的数量
    val violationDF = snapShotsDF.join(aliveUserDF,"user_id").filter($"violations_label" === 0)
      .select("user_id").groupBy("user_id").agg(count("user_id").as("violations_count"))
    val sexualDF = snapShotsDF.join(aliveUserDF,"user_id").filter($"violations_label" === 1)
      .select("user_id").groupBy("user_id").agg(count("user_id").as("sexual_count"))
    val normalDF = snapShotsDF.join(aliveUserDF,"user_id").filter($"violations_label" === 2)
      .select("user_id").groupBy("user_id").agg(count("user_id").as("normal_count"))

    // 对截图记录中的场景、视频类型、匹配model进行统计
    val pageDF = snapShotsDF.join(aliveUserDF,"user_id")
      .select("user_id","page").groupBy("user_id")
      .agg(sum("page").as("page_sum"))
    val modelDF = snapShotsDF.join(aliveUserDF,"user_id")
      .select("user_id","model").groupBy("user_id")
      .agg(sum("model").as("model_sum"))
    val videoTypeDF = snapShotsDF.join(aliveUserDF,"user_id")
      .select("user_id","video_type").groupBy("user_id")
      .agg(sum("video_type").as("video_type_sum"))


    //合并截图统计结果
    val snapShotsUnionDF = snapShotsDF.join(violationDF,Seq("user_id"),"left_outer")
      .join(sexualDF,Seq("user_id"),"left_outer")
      .join(normalDF,Seq("user_id"),"left_outer")
      .join(pageDF,Seq("user_id"),"left_outer")
      .join(modelDF,Seq("user_id"),"left_outer")
      .join(videoTypeDF,Seq("user_id"),"left_outer")


    // 产生标签函数（取决于用户近期视频截图中违规截图数占总截图数的比例，以及用户本身色情属性计算决定）
    val getLabelUDF = udf((violations_count:Int, sexual_count:Int, normal_count:Int) =>
    {
      var res:Int = -1
      val rate: Float = (violations_count * 2 + sexual_count * 1) / ( (violations_count+sexual_count+normal_count)* 2)
      if(rate>0.5){
        res = 1
      }else{
        res = 0
      }
    })

    // 产生标签
    val labelDF = videoRecordDF.join(snapShotsUnionDF, "user_id")
      .withColumn("label", getLabelUDF($"violations_count", $"sexual_count", $"normal_count"))
      .filter($"label" =!= -1)
      .groupBy("user_id").agg(max("label").as("label"))

    /***
     * 特征合并部分
     ***/

    // 男性用户对女性的行为合并
    val boyToGirlProfileDF = labelDF
      .join(sendGiftBoyToGirlDF, Seq("user_id", "target_user_id"), "left_outer")
      .join(sendFriendVideoCallBoyToGirlDF, Seq("user_id", "target_user_id"), "left_outer")
      .join(sendFriendVideoRecordBoyToGirlDF, Seq("user_id", "target_user_id"), "left_outer")
      .join(sendGoddessWallCallBoyToGirlDF, Seq("user_id", "target_user_id"), "left_outer")
      .join(matchVideoRecordBoyToGirlDF, Seq("user_id", "target_user_id"), "left_outer")
      .join(sendTextChatBoyToGirlDF, Seq("user_id", "target_user_id"), "left_outer")
      .join(sendEffectBoyToGirlDF, Seq("user_id", "target_user_id"), "left_outer")
      .join(sendVoiceBoyToGirlDF, Seq("user_id", "target_user_id"), "left_outer")
      .join(addFriendBoyToGirlDF, Seq("user_id", "target_user_id"), "left_outer")
      .join(likeBoyToGirlDF, Seq("user_id", "target_user_id"), "left_outer")
      .join(clickAvatorBoyToGirlDF, Seq("user_id", "target_user_id"), "left_outer")
      .join(collectBoyToGirlDF, Seq("user_id", "target_user_id"), "left_outer")
      .join(onlineRemindBoyToGirlDF, Seq("user_id", "target_user_id"), "left_outer")
    // 完整特征合并
    val resDF = boyToGirlProfileDF
      .join(boyProfileDF, Seq("user_id"), "left_outer")
      .join(girlProfileDF, Seq("target_user_id"), "left_outer")

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

    // 将数据进行拆分。分为训练集和测试集
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

    // deleteFile(spark, lr_model_savepath, today, historyOffsetDay)
    model.save(lr_model_savepath + today)
    println(s"Model has been saved to ${lr_model_savepath + today} !")

  }
}
