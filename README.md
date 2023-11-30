# CS337_Project
## Anish Kulkarni, Ashwin Abraham, Cheshta Damor, Parth Pujari, <\br>
First run
```
pip install -r requirements.txt
```
To train the ddpm model, i.e., to run the file ```ddpm.py``` run the command (this also shows the generated images after each epoch):
```
python3 ddpm.py --train --epochs $EPOCHS
```
To run ```gan_ddp.py``` run the command
```
python3 gan_ddp.py
```
To run the ddgan code from the paper run the commands in ```readme.md``` in the ddgan folder
