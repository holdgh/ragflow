#!/bin/bash
# 在dockerfile中已经将service_conf.yaml.template拷贝到conf目录下
# replace env variables in the service_conf.yaml file
# 删除已存在的配置文件
rm -rf /ragflow/conf/service_conf.yaml
#逐行读取service_conf.yaml.template，将其中的环境变量占位符替换为相应的环境变量，并输出替换后的内容到service_conf.yaml
while IFS= read -r line || [[ -n "$line" ]]; do
    # Use eval to interpret the variable with default values
    eval "echo \"$line\"" >> /ragflow/conf/service_conf.yaml
done < /ragflow/conf/service_conf.yaml.template

# unset http proxy which maybe set by docker daemon
export http_proxy=""; export https_proxy=""; export no_proxy=""; export HTTP_PROXY=""; export HTTPS_PROXY=""; export NO_PROXY=""

/usr/sbin/nginx

export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/

PY=python3
# 如果$WS长度为0， 或者值小于1，则赋值为1
if [[ -z "$WS" || $WS -lt 1 ]]; then
  WS=1
fi

function task_exe(){
#  死循环？
    while [ 1 -eq 1 ];do
      $PY rag/svr/task_executor.py $1;
    done
}

for ((i=0;i<WS;i++))
do
  task_exe  $i &
done
#  死循环？
while [ 1 -eq 1 ];do
    $PY api/ragflow_server.py
done

wait;
