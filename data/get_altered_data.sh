# !/bin/bash
filename="Tiny-ImageNet-C.tar"
fileid="1qosir0sn9ulQvBqMpYjL0IgpA_f7vD6T"
query=`curl -c ./cookie.txt -s -L "https://drive.google.com/uc?export=download&id=${fileid}" | pup 'a#uc-download-link attr{href}' | sed -e 's/amp;//g'`
curl -b ./cookie.txt -L -o ${filename} "https://drive.google.com${query}"
tar -xvf Tiny-ImageNet-C.tar
