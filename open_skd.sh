python main.py /data/wzy/CLIPood/CoOp/data/ --data Caltech101          --task open_class --n-shot 16 --batch-size 36 --lr 5e-6 --epochs 10 --alpha 0.8  #1
python main.py /data/wzy/CLIPood/CoOp/data/ --data DescribableTextures --task open_class --n-shot 16 --batch-size 36 --lr 5e-6 --epochs 10 --alpha 0.8 --position front  #1
python main.py /data/wzy/CLIPood/CoOp/data/ --data EuroSAT             --task open_class --n-shot 16 --batch-size 36 --lr 5e-6 --epochs 8  --alpha 0.8  #1
python main.py /data/wzy/CLIPood/CoOp/data/ --data FGVCAircraft        --task open_class --n-shot 16 --batch-size 36 --lr 5e-6 --epochs 20 --alpha 0.8 --position middle  #10
python main.py /data/wzy/CLIPood/CoOp/data/ --data Food101             --task open_class --n-shot 16 --batch-size 36 --lr 5e-6 --epochs 8  --alpha 0.8 --position middle  #1
python main.py /data/wzy/CLIPood/CoOp/data/ --data OxfordFlowers       --task open_class --n-shot 16 --batch-size 36 --lr 5e-6 --epochs 8  --alpha 0.8  #1
python main.py /data/wzy/CLIPood/CoOp/data/ --data OxfordPets          --task open_class --n-shot 16 --batch-size 36 --lr 5e-6 --epochs 8  --alpha 0.8 --position middle  #1
python main.py /data/wzy/CLIPood/CoOp/data/ --data StanfordCars        --task open_class --n-shot 16 --batch-size 36 --lr 4e-6 --epochs 20 --alpha 0.8  #5
python main.py /data/wzy/CLIPood/CoOp/data/ --data SUN397              --task open_class --n-shot 16 --batch-size 36 --lr 5e-6 --epochs 28 --alpha 0.8  #5
python main.py /data/wzy/CLIPood/CoOp/data/ --data UCF101              --task open_class --n-shot 16 --batch-size 36 --lr 5e-6 --epochs 16 --alpha 0.8  #2
