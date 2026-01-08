# testForFile.sh 
# test if the file exists
# https://linuxize.com/post/bash-check-if-file-exists/
# 

FILE=$1
if [ -f "$FILE" ]; then
   echo "$FILE exists."
   exit 0
else
   echo "$FILE does not exist."
   exit 1
fi

