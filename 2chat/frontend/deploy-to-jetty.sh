#!/bin/sh
dir="$(pwd)"
rm -r build/
npm run build
cp -r /home/w/IdeaProjects/tochat/src/main/webapp/WEB-INF /home/w/IdeaProjects/tochat/src/main/
cp -r /home/w/IdeaProjects/tochat/target/chat/WEB-INF /home/w/IdeaProjects/tochat/target
cp -r /home/w/IdeaProjects/tochat/target/chat/META-INF /home/w/IdeaProjects/tochat/target
cd /home/w/IdeaProjects/tochat/src/main/webapp/
rm -r *
cd /home/w/IdeaProjects/tochat/target/chat/
rm -r *
cd "$dir"
cd ./build
echo "$(pwd)"
cp -r ./* /home/w/IdeaProjects/tochat/src/main/webapp/
cp -r ./* /home/w/IdeaProjects/tochat/target/chat/
mv /home/w/IdeaProjects/tochat/src/main/WEB-INF /home/w/IdeaProjects/tochat/src/main/webapp/WEB-INF
mv /home/w/IdeaProjects/tochat/target/META-INF /home/w/IdeaProjects/tochat/target/chat/META-INF
mv /home/w/IdeaProjects/tochat/target/WEB-INF /home/w/IdeaProjects/tochat/target/chat/WEB-INF
cd ..
echo "Completed"