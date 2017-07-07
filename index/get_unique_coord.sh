cat indices.txt > tmp.txt
sort tmp.txt | uniq > indices.txt
rm tmp.txt
