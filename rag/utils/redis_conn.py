import logging
import json

import valkey as redis
from rag import settings
from rag.utils import singleton


class Payload:
    def __init__(self, consumer, queue_name, group_name, msg_id, message):
        self.__consumer = consumer
        self.__queue_name = queue_name
        self.__group_name = group_name
        self.__msg_id = msg_id
        self.__message = json.loads(message['message'])

    def ack(self):
        # 消费者向队列回复"我收到了数据【msg_id】，我已经处理了数据"
        try:
            self.__consumer.xack(self.__queue_name, self.__group_name, self.__msg_id)
            return True
        except Exception as e:
            logging.warning("[EXCEPTION]ack" + str(self.__queue_name) + "||" + str(e))
        return False

    def get_message(self):
        return self.__message


@singleton
class RedisDB:
    def __init__(self):
        self.REDIS = None
        self.config = settings.REDIS
        self.__open__()

    def __open__(self):
        try:
            self.REDIS = redis.StrictRedis(host=self.config["host"].split(":")[0],
                                     port=int(self.config.get("host", ":6379").split(":")[1]),
                                     db=int(self.config.get("db", 1)),
                                     password=self.config.get("password"),
                                     decode_responses=True)
        except Exception:
            logging.warning("Redis can't be connected.")
        return self.REDIS

    def health(self):

        self.REDIS.ping()
        a, b = 'xx', 'yy'
        self.REDIS.set(a, b, 3)

        if self.REDIS.get(a) == b:
            return True

    def is_alive(self):
        return self.REDIS is not None

    def exist(self, k):
        if not self.REDIS: return
        try:
            return self.REDIS.exists(k)
        except Exception as e:
            logging.warning("[EXCEPTION]exist" + str(k) + "||" + str(e))
            self.__open__()

    def get(self, k):
        if not self.REDIS: return
        try:
            return self.REDIS.get(k)
        except Exception as e:
            logging.warning("[EXCEPTION]get" + str(k) + "||" + str(e))
            self.__open__()

    def set_obj(self, k, obj, exp=3600):
        try:
            self.REDIS.set(k, json.dumps(obj, ensure_ascii=False), exp)
            return True
        except Exception as e:
            logging.warning("[EXCEPTION]set_obj" + str(k) + "||" + str(e))
            self.__open__()
        return False

    def set(self, k, v, exp=3600):
        # 字符串类型保存操作
        try:
            self.REDIS.set(k, v, exp)
            return True
        except Exception as e:
            logging.warning("[EXCEPTION]set" + str(k) + "||" + str(e))
            self.__open__()
        return False

    def sadd(self, key: str, member: str):
        # 集合set类型保存操作
        try:
            self.REDIS.sadd(key, member)
            return True
        except Exception as e:
            logging.warning("[EXCEPTION]sadd" + str(key) + "||" + str(e))
            self.__open__()
        return False

    def srem(self, key: str, member: str):
        try:
            self.REDIS.srem(key, member)
            return True
        except Exception as e:
            logging.warning("[EXCEPTION]srem" + str(key) + "||" + str(e))
            self.__open__()
        return False

    def smembers(self, key: str):
        try:
            res = self.REDIS.smembers(key)
            return res
        except Exception as e:
            logging.warning("[EXCEPTION]smembers" + str(key) + "||" + str(e))
            self.__open__()
        return None

    def zadd(self, key: str, member: str, score: float):
        # 有序集合zset类型保存操作
        try:
            self.REDIS.zadd(key, {member: score})
            return True
        except Exception as e:
            logging.warning("[EXCEPTION]zadd" + str(key) + "||" + str(e))
            self.__open__()
        return False

    def zcount(self, key: str, min: float, max: float):
        try:
            res = self.REDIS.zcount(key, min, max)
            return res
        except Exception as e:
            logging.warning("[EXCEPTION]spopmin" + str(key) + "||" + str(e))
            self.__open__()
        return 0

    def zpopmin(self, key: str, count: int):
        try:
            res = self.REDIS.zpopmin(key, count)
            return res
        except Exception as e:
            logging.warning("[EXCEPTION]spopmin" + str(key) + "||" + str(e))
            self.__open__()
        return None

    def zrangebyscore(self, key: str, min: float, max: float):
        """
        1、zrange( name, start, end, desc=False, withscores=False, score_cast_func=float)
            按照索引范围获取name对应的有序集合的元素
            参数：

                name - redis的name
                start - 有序集合索引起始位置（非分数）
                end - 有序集合索引结束位置（非分数）
                desc - 排序规则，默认按照分数从小到大排序
                withscores - 是否获取元素的分数，默认只获取元素的值
                score_cast_func - 对分数进行数据转换的函数
        2、zrangebyscore(name, min, max, start=None, num=None, withscores=False, score_cast_func=float)
            按照分数范围【从小到大】获取name对应的有序集合的元素
        3、zrevrangebyscore(name, max, min, start=None, num=None, withscores=False, score_cast_func=float)
            按照分数范围获取有序集合的元素并排序（默认从大到小排序）
        4、zscan(name, cursor=0, match=None, count=None, score_cast_func=float)
            获取所有元素–默认按照分数顺序【从小到大】排序
        """
        try:
            # 从小到大
            res = self.REDIS.zrangebyscore(key, min, max)
            return res
        except Exception as e:
            logging.warning("[EXCEPTION]srangebyscore" + str(key) + "||" + str(e))
            self.__open__()
        return None

    def transaction(self, key, value, exp=3600):
        try:
            pipeline = self.REDIS.pipeline(transaction=True)
            pipeline.set(key, value, exp, nx=True)
            pipeline.execute()
            return True
        except Exception as e:
            logging.warning("[EXCEPTION]set" + str(key) + "||" + str(e))
            self.__open__()
        return False

    def queue_product(self, queue, message, exp=settings.SVR_QUEUE_RETENTION) -> bool:
        for _ in range(3):
            try:
                payload = {"message": json.dumps(message)}
                # redis默认在执行每次请求都会创建（连接池申请连接）和断开（归还连接池）一次连接操作，如果想要在一次请求中指定多个命令，则可以使用pipline实现一次请求指定多个命令，并且默认情况下一次pipline 是原子性操作。
                # 管道（pipeline）是redis在提供单个请求中缓冲多条服务器命令的基类的子类。它通过减少服务器-客户端之间反复的TCP数据库包，从而大大提高了执行批量命令的功能
                pipeline = self.REDIS.pipeline()
                pipeline.xadd(queue, payload)
                #pipeline.expire(queue, exp)
                pipeline.execute()
                return True
            except Exception:
                logging.exception("producer" + str(queue) + " got exception")
        return False

    def queue_consumer(self, queue_name, group_name, consumer_name, msg_id=b">") -> Payload:
        try:
            # 指定队列不存在指定组时，则创建组
            group_info = self.REDIS.xinfo_groups(queue_name)
            if not any(e["name"] == group_name for e in group_info):
                self.REDIS.xgroup_create(
                    queue_name,
                    group_name,
                    id="0",
                    mkstream=True
                )
            args = {
                "groupname": group_name,
                "consumername": consumer_name,
                "count": 1,
                "block": 10000,
                "streams": {queue_name: msg_id},
            }
            # 读取指定队列的指定组的指定消费者的数据
            messages = self.REDIS.xreadgroup(**args)
            if not messages:
                # 读取不到数据时，直接返回none
                return None
            # 从读取的数据中构造负载并返回
            stream, element_list = messages[0]
            msg_id, payload = element_list[0]
            res = Payload(self.REDIS, queue_name, group_name, msg_id, payload)
            return res
        except Exception as e:
            if 'key' in str(e):
                pass
            else:
                logging.exception("consumer: " + str(queue_name) + " got exception")
        return None

    def get_unacked_for(self, consumer_name, queue_name, group_name):
        """
        功能：从指定队列的指定组的指定消费者的缓存数据中取出一个数据，作为等待数据，构造成负载数据并返回
        """
        try:
            # 查询指定队列的组列表
            group_info = self.REDIS.xinfo_groups(queue_name)
            # 筛选指定组名的组，不存在则直接返回
            if not any(e["name"] == group_name for e in group_info):
                return
            # 获取指定组的特定消费者的一个等待数据？
            pendings = self.REDIS.xpending_range(queue_name, group_name, min=0, max=10000000000000, count=1, consumername=consumer_name)
            # 不存在等待数据，则直接返回
            if not pendings: return
            # 获取等待数据的msg_id
            msg_id = pendings[0]["message_id"]
            # 从指定队列获的msg_id处获取1个msg
            msg = self.REDIS.xrange(queue_name, min=msg_id, count=1)
            _, payload = msg[0]
            # 构造并返回负载数据
            return Payload(self.REDIS, queue_name, group_name, msg_id, payload)
        except Exception as e:
            if 'key' in str(e):
                return
            logging.exception("xpending_range: " + consumer_name + " got exception")
            self.__open__()

    def queue_info(self, queue, group_name) -> dict:
        # 从redis查询指定队列queue的指定组group_name，不存在则返回none
        for _ in range(3):
            try:
                groups = self.REDIS.xinfo_groups(queue)
                for group in groups:
                    if group["name"] == group_name:
                        return group
            except Exception:
                logging.exception("queue_length" + str(queue) + " got exception")
        return None

    def queue_head(self, queue) -> int:
        for _ in range(3):
            try:
                ent = self.REDIS.xrange(queue, count=1)
                return ent[0]
            except Exception:
                logging.exception("queue_head" + str(queue) + " got exception")
        return 0

REDIS_CONN = RedisDB()
