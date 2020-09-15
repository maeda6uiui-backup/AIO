import os
import torch

if __name__=="__main__":
    dir1="/home/acc12147kk/EncodedCacheNICT2/Train_FirstHalf"
    dir2="/home/acc12147kk/EncodedCacheNICT2/Train_SecondHalf"
    save_dir="/home/acc12147kk/EncodedCacheNICT2/Train"

    input_ids_1_filepath=os.path.join(dir1,"input_ids.pt")
    attention_mask_1_filepath=os.path.join(dir1,"attention_mask.pt")
    token_type_ids_1_filepath=os.path.join(dir1,"token_type_ids.pt")
    labels_1_filepath=os.path.join(dir1,"labels.pt")

    input_ids_1=torch.load(input_ids_1_filepath)
    attention_mask_1=torch.load(attention_mask_1_filepath)
    token_type_ids_1=torch.load(token_type_ids_1_filepath)
    labels_1=torch.load(labels_1_filepath)

    input_ids_2_filepath=os.path.join(dir2,"input_ids.pt")
    attention_mask_2_filepath=os.path.join(dir2,"attention_mask.pt")
    token_type_ids_2_filepath=os.path.join(dir2,"token_type_ids.pt")
    labels_2_filepath=os.path.join(dir2,"labels.pt")

    input_ids_2=torch.load(input_ids_2_filepath)
    attention_mask_2=torch.load(attention_mask_2_filepath)
    token_type_ids_2=torch.load(token_type_ids_2_filepath)
    labels_2=torch.load(labels_2_filepath)

    input_ids=torch.cat([input_ids_1,input_ids_2],dim=0)
    attention_mask=torch.cat([attention_mask_1,attention_mask_2],dim=0)
    token_type_ids=torch.cat([token_type_ids_1,token_type_ids_2],dim=0)
    labels=torch.cat([labels_1,labels_2],dim=0)

    os.makedirs(save_dir,exist_ok=True)
    input_ids_filepath=os.path.join(save_dir,"input_ids.pt")
    attention_mask_filepath=os.path.join(save_dir,"attention_mask.pt")
    token_type_ids_filepath=os.path.join(save_dir,"token_type_ids.pt")
    labels_filepath=os.path.join(save_dir,"labels.pt")

    torch.save(input_ids,input_ids_filepath)
    torch.save(attention_mask,attention_mask_filepath)
    torch.save(token_type_ids,token_type_ids_filepath)
    torch.save(labels,labels_filepath)
