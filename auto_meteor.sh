prefix=$1
suffix=$2
path='./res/'
system_out="_system_out_"
reference_out="_groundtruth_"
java -Xmx2G -jar ../eval_script/meteor-1.5/meteor-1.5.jar $path$prefix$system_out$suffix $path$prefix$reference_out$suffix -l en -norm -r 1 | tail -1

