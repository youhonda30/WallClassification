import argparse as ap
import os
import shutil
#重複ファイルを削除するスクリプト。inputの中にあるファイルと同名のファイルをtargetから削除する

#root_dir下のファイルパスを全列挙
def walk(root_dir):
    for root,_,files in os.walk(root_dir):
        for file in files:
            yield os.path.join(root,file)

if __name__ == "__main__": 
    parser = ap.ArgumentParser()
    parser.add_argument("target", help="削除する重複ファイルが入っているディレクトリ")
    parser.add_argument("-l","--length", help="削除するファイル名の長さの閾値", default = 200, type = int)
    args = parser.parse_args()
    target = args.target
    for path in [os.path.relpath(path,target) for path in walk(target)]:
        if len(path) > args.length:
            print("remove:",path)
            shutil.os.remove(os.path.join(target,path))