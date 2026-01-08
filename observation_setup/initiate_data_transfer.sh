start_time=$1

date
echo Start observation

sleep $start_time

touch ~/BOOKEEP/Trigger.txt

date

echo End observation
