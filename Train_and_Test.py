#%%
import matplotlib.pyplot as plt
import numpy as np
import torch
#%%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

'''train function define'''
def train(model, data_dl, scheduler, start):
    model.train()
    running_loss = 0.0
    running_psnr = 0.0
    running_ssim = 0.0
    for ba, (noisy, clean) in enumerate(data_dl):
        image = noisy.to(device)
        label = clean.to(device)
        optimizer.zero_grad()

        # forward
        outputs = model(image.float())
        loss = criterion(outputs, label.float())
        # backward
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * noisy.shape[0]
        batch_psnr = PSNR(outputs, label)
        batch_ssim = SSIMbatch(outputs, label)
        running_psnr += batch_psnr
        running_ssim += batch_ssim

        if ((ba+1) % 50) == 0:
            endbatch = time.time()
            print('{}th batch time : {:.1f}m {:.1f}s'.format(ba+1, (endbatch-start)//60, (endbatch-start)%60) )
        if ((ba+1) % 500) == 0: 
            resultimage = outputs[0].detach().permute(1,2,0).cpu().squeeze().numpy()
            origin = noisy[0].squeeze().numpy()
            origin = 0.5*origin + 0.5
            origin *= 255

            resultimage = 0.5*resultimage + 0.5
            resultimage = np.clip(resultimage*255, 0, 255).astype('uint8')
            

            plt.figure(figsize=(30,15))
            plt.subplot(1,2,1)
            plt.imshow(origin, cmap = 'gray')
            plt.title('noisy')
            plt.subplot(1,2,2)
            plt.imshow(resultimage, cmap = 'gray')
            plt.title('clean')
            plt.show()

    scheduler.step()
            



    final_loss = running_loss / len(data_dl.dataset)
    # final_psnr = running_psnr / int(len(trainset)/data_dl.batch_size)
    final_psnr = running_psnr / (ba+1)
    final_ssim = running_ssim / len(data_dl.dataset)

    return final_loss, final_psnr, final_ssim

# %%
'''validation function define'''
def test(model, data_dl):
    print('test began')
    # epoch는 이미지를 저장할때, 이미지의 이름으로 사용 (미적용)
    model.eval()
    running_loss = 0.0
    running_psnr = 0.0
    running_ssim = 0.0
    with torch.no_grad():
        for ba, (noisy, clean) in enumerate(data_dl):
            image = noisy.float().to(device)
            label = clean.float().to(device)

            outputs = model(image)
            loss = criterion(outputs, label)

            running_loss += loss.item() * noisy.shape[0]
            batch_psnr = PSNR(outputs, label)
            batch_ssim = SSIMbatch(outputs, label)
            running_psnr += batch_psnr
            running_ssim += batch_ssim

        # outputs = outputs.cpu()
        # save_image(outputs, f'/content/outputs/{epoch}.png')
    
    final_loss = running_loss / len(data_dl.dataset)
    # final_psnr = running_psnr / int(len(testset)/data_dl.batch_size)
    final_psnr = running_psnr / (ba+1)
    final_ssim = running_ssim / len(data_dl.dataset)

    return final_loss, final_psnr, final_ssim
