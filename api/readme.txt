后端api服务
类似于Java后端服务架构：数据库【db目录】-工具类【utils目录】-service【db.services目录】-controller【api目录的*_app.py和sdk包】

db/db_models.py定义了数据表实体类和基础数据库类【用以封装基础的数据库操作，其他数据表实体类继承自该类，以实现相应实体的基础数据库操作】