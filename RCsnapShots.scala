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
import org.apache.spark.ml.feature.{OneHotEncoder, OneHotEncoderEstimator, QuantileDiscretizer,Bucketizer, StringIndexer, VectorAssembler, VectorIndexer}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{udf, _}
import org.apache.spark.sql.types.IntegerType

object RCsnapShots extends TaskSchedule{
  def main(args:Array[String]) {

    val spark = SpeckConfig.getSpeckConfigInstance(this.getClass.getSimpleName.filter(!_.equals('$')))
    import spark.implicits._
    import spark.sql

    /** *
     * 1
     * **/

      //从出现色情截图开始，到挂断为止，色情聊天时长统计
    val shotsDF1 = sql("select location,room_id from rc_live_chat_statistics.rc_video_snapshots where dt>'2019-12-11' and violations_label=0").groupBy("room_id").agg(min("location").as("min"), max("location").as("max")).withColumn("duration", $"max" - $"min")

    val recordDF = sql("select room_id,video_time  /  1000 video_time from rc_live_chat_statistics.rc_video_record where dt>'2019-12-11'").distinct()

    //  val sumDF = shotsDF.groupBy("duration").agg(count("room_id").as("Nums")).orderBy(desc("Nums"))

    // 合并shotsDF & recordDF
    val statisticsDF = shotsDF1.join(recordDF, Seq("room_id"), "left_outer").withColumn("violation_duration", $"video_time" - $"min").orderBy(desc("violation_duration"))

    //对duration类进行分桶
    val splits = Array(Double.NegativeInfinity, 0, 1, 5, 10, 15, 20, 30, 40, 60, 90, 120, 180, 300, 500, 1000, 2000, 4000, Double.PositiveInfinity)
    val quantileTransformers = new Bucketizer().setInputCol("violation_duration").setOutputCol("violation_duration_class").setSplits(splits).setHandleInvalid("skip")
    val resDF = quantileTransformers.transform(statisticsDF)

    //筛选最终结果
    val data1 = resDF.select("violation_duration_class", "room_id").groupBy("violation_duration_class").agg(count("room_id").as("Nums")).orderBy(desc("violation_duration_class"))
    //保存
    data1.coalesce(1).write.mode("Append").csv("/home/hadoop/job-dir/aoxiang/sample_file.csv")

    /** *
     * 2
     * **/

    //所有色情截图时间位置统计
    val locationDF = sql("select location from rc_live_chat_statistics.rc_video_snapshots where dt>'2019-12-11' and violations_label=0").groupBy("location").agg(count("location").as("Nums")).orderBy(desc("Nums"))

    val splitss = Array(0, 1, 4, 6, 10, 15, 20, 25, 30, 60, 100, Double.PositiveInfinity)
    val bucketizerTransformers = new Bucketizer().setInputCol("location").setOutputCol("location_class").setSplits(splitss).setHandleInvalid("skip")
    val data2 = bucketizerTransformers.transform(locationDF).select("location_class", "Nums").groupBy("location_class").agg(sum("Nums").as("Sum")).orderBy(desc("location_class"))

    /** *
     * 3
     * **/

      // 对每个用户，基于其历史色情截图时间，给出5个预测截图时间点
    val userShotsDF = sql("select user_id,location,room_id from rc_live_chat_statistics.rc_video_snapshots where dt>'2019-12-11' and violations_label=0").groupBy("user_id").agg(count("room_id").as("Nums"),min("location").as("min"), max("location").as("max")).withColumn("gap", $"max" - $"min").distinct()

    val snDF = sql("select user_id,location from rc_live_chat_statistics.rc_video_snapshots where dt>'2019-12-11' and violations_label=0").groupBy("user_id").agg(collect_list("location").as("collection"))

    def abs(n: Int): Int =
      if (n > 0) n else -n

    val sortingUDF = udf((collections:Seq[Int]) => {
      val res = scala.collection.mutable.ListBuffer[Int]()
      val clo = collections.toList.map((_,1)).groupBy(_._1).mapValues(_.size).toList.sortBy(-_._2)
      if(clo(0)._1>6){
        res += 3
      }
      res += clo(0)._1
      var i = 1
      while(i<clo.length && res.length<=5){
        if(abs(clo(i)._1 - res.last)>=10){
          res += clo(i)._1
        }
        i += 1
      }

      while(res.length<=5){
        res += res.last + 10
      }

      res.toList.sorted
    })

    val resDF1 = snDF.na.fill(3).withColumn("predictions",sortingUDF($"collection"))

    /** *
     * 4
     * **/
     //统计视频时长
    val timeDF = sql("select video_time/1000 video_time from rc_live_chat_statistics.rc_video_record where dt>'2019-12-11'")

    val splitsss = Array(0, 1, 4, 6, 10, 15, 20, 25, 30, 60, 100, Double.PositiveInfinity)
    val bucketizerTransformers1 = new Bucketizer().setInputCol("video_time").setOutputCol("video_duration").setSplits(splitsss).setHandleInvalid("skip")
    val data3 = bucketizerTransformers1.transform(timeDF).select( "video_duration").groupBy("video_duration").agg(count("video_duration").as("Nums")).orderBy(desc("video_duration"))

    /** *
     * 5
     * **/

    //将按照固定3、6秒截图的用户与5%均匀截图的测试用户区分开
    val chosenUserDF = sql("select user_id, location, room_id from rc_live_chat_statistics.rc_video_snapshots where dt>'2019-12-11'")
      .groupBy("user_id").agg(min("location").as("min"), max("location").as("max"))
      .filter($"max">6).filter($"min"===5)
      .select("user_id")


    //统计5%用户的色情截图时间
    val shotsDF = sql("select user_id, location, room_id from rc_live_chat_statistics.rc_video_snapshots where dt>'2019-12-11' and violations_label=0")
      .groupBy("user_id").agg(min("location").as("min"), max("location").as("max"),count("location").as("Nums"))
      .filter($"max">6).filter($"min"===5)
      .withColumn("duration", $"max" - $"min")
      .join(chosenUserDF,Seq("user_id"),"left_outer").na.fill(0)








  }
}
