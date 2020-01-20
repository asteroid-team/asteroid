import torch
from torch import nn
import torchvision
import torch.nn.functional as F 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_cRM(Y,S):
    '''
    :param Y: mixed/noisy stft  (N,H,W,C)
    :param S: clean stft (N,H,W,C)
    :return: structed cRM
    '''
    M = torch.zeros(Y.shape)
    epsilon = 1e-8
    # real part
    M_real = (Y[...,0] * S[...,0]) + (Y[...,1] * S[...,1])
    square_real = (Y[...,0] ** 2) + (Y[...,1] ** 2)
    M_real = (M_real / (square_real+epsilon))
    M[...,0] = M_real
    # imaginary part
    M_img = (Y[...,0] * S[...,1]) - (Y[...,1] * S[...,0])
    square_img = (Y[...,0] ** 2) + (Y[...,1] ** 2)
    M_img = (M_img / (square_img + epsilon))
    M[...,1] = M_img
    return M

def cRM_tanh_compress(M,K=10,C=0.1):
    '''
    Recall that the irm takes on vlaues in the range[0,1],compress the cRM with hyperbolic tangent
    :param M: crm (298,257,2)
    :param K: parameter to control the compression
    :param C: parameter to control the compression
    :return crm: compressed crm
    '''

    numerator = 1-torch.exp(-C*M)
    numerator[numerator == inf] = 1
    numerator[numerator == -inf] = -1
    denominator = 1+torch.exp(-C*M)
    denominator[denominator == inf] = 1
    denominator[denominator == -inf] = -1
    crm = K * (numerator / denominator)

    return crm

def cRM_tanh_recover(O,K=10,C=0.1):
    '''
    :param O: predicted compressed crm
    :param K: parameter to control the compression
    :param C: parameter to control the compression
    :return M : uncompressed crm
    '''

    numerator = K-O
    denominator = K+O
    M = -((1.0/C) * torch.log((numerator / denominator)))

    return M


def fast_cRM(Fclean,Fmix,K=10,C=0.1):
    '''
    :param Fmix: mixed/noisy stft
    :param Fclean: clean stft
    :param K: parameter to control the compression
    :param C: parameter to control the compression
    :return crm: compressed crm
    '''
    M = generate_cRM(Fmix,Fclean)
    crm = cRM_tanh_compress(M,K,C)
    return crm

def fast_icRM(Y,crm,K=10,C=0.1):
    '''
    :param Y: mixed/noised stft
    :param crm: DNN output of compressed crm
    :param K: parameter to control the compression
    :param C: parameter to control the compression
    :return S: clean stft
    '''
    M = cRM_tanh_recover(crm,K,C)
    S = torch.zeros(M.shape)
    S[:, 0, ...] = (M[:, 0, ...] * Y[:, 0, ...]) - (M[:, 1, ...] * Y[:, 1, ...])
    S[:, 1, ...] = (M[:, 0,...] * Y[:, 1, ...]) + (M[:, 1, ...] * Y[:, 0, ...])
    return S


class Audio_Model(nn.Module):
    
    def __init__(self):
        super(Audio_Model, self).__init__()
       
        #Audio model layers , name of layers as per table 1 given in paper.
        
        self.conv1 = nn.Conv2d(2,96,kernel_size = (1,7),padding = self.get_padding((1,7),(1,1)), dilation = (1,1))
      
        self.conv2 = nn.Conv2d(96,96,kernel_size = (7,1),padding = self.get_padding((7,1),(1,1)), dilation = (1,1))
      
        self.conv3 = nn.Conv2d(96,96,kernel_size = (5,5),padding = self.get_padding((5,5),(1,1)), dilation = (1,1))
      
        self.conv4 = nn.Conv2d(96,96,kernel_size = (5,5),padding = self.get_padding((5,5),(2,1)), dilation = (2,1))
      
        self.conv5 = nn.Conv2d(96,96,kernel_size = (5,5),padding = self.get_padding((5,5),(4,1)), dilation = (4,1))
      
        self.conv6 = nn.Conv2d(96,96,kernel_size = (5,5),padding = self.get_padding((5,5),(8,1)), dilation = (8,1))
      
        self.conv7 = nn.Conv2d(96,96,kernel_size = (5,5),padding = self.get_padding((5,5),(16,1)), dilation = (16,1))
      
        self.conv8 = nn.Conv2d(96,96,kernel_size = (5,5),padding = self.get_padding((5,5),(32,1)), dilation = (32,1))
        
        self.conv9 = nn.Conv2d(96,96,kernel_size = (5,5),padding = self.get_padding((5,5),(1,1)), dilation = (1,1))
                               
        self.conv10 = nn.Conv2d(96,96,kernel_size = (5,5),padding = self.get_padding((5,5),(2,2)), dilation = (2,2))
        
        self.conv11 = nn.Conv2d(96,96,kernel_size = (5,5),padding = self.get_padding((5,5),(4,4)), dilation = (4,4))
        
        self.conv12 = nn.Conv2d(96,96,kernel_size = (5,5),padding = self.get_padding((5,5),(8,8)), dilation = (8,8))
        
        self.conv13 = nn.Conv2d(96,96,kernel_size = (5,5),padding = self.get_padding((5,5),(16,16)), dilation = (16,16))
        
        self.conv14 = nn.Conv2d(96,96,kernel_size = (5,5),padding = self.get_padding((5,5),(32,32)), dilation = (32,32))
        
        self.conv15 = nn.Conv2d(96,8,kernel_size = (1,1),padding = self.get_padding((1,1),(1,1)), dilation = (1,1))
        
        #Batch normalization layers
        
        self.batch_norm1 = nn.BatchNorm2d(96)
        self.batch_norm2 = nn.BatchNorm2d(96)
        self.batch_norm3 = nn.BatchNorm2d(96)
        self.batch_norm4 = nn.BatchNorm2d(96)
        self.batch_norm5 = nn.BatchNorm2d(96)
        self.batch_norm6 = nn.BatchNorm2d(96)
        self.batch_norm7 = nn.BatchNorm2d(96)
        self.batch_norm8 = nn.BatchNorm2d(96)
        self.batch_norm9 = nn.BatchNorm2d(96)
        self.batch_norm10 = nn.BatchNorm2d(96)
        self.batch_norm11 = nn.BatchNorm2d(96)
        self.batch_norm11 = nn.BatchNorm2d(96)
        self.batch_norm12 = nn.BatchNorm2d(96)
        self.batch_norm13 = nn.BatchNorm2d(96)
        self.batch_norm14 = nn.BatchNorm2d(96)
        self.batch_norm15 = nn.BatchNorm2d(8)
        
    def get_padding(self,kernel_size,dilation):
        padding = (((dilation[0])*(kernel_size[0]-1))//2,((dilation[1])*(kernel_size[1]-1))//2)
        return padding
        
    def forward(self,input_audio):
        # input audio will be (2,298,257)
        
        output_layer = F.relu(self.batch_norm1(self.conv1(input_audio)))
        #print (output_layer.shape)
        output_layer = F.relu(self.batch_norm2(self.conv2(output_layer)))
        #print (output_layer.shape)
        output_layer = F.relu(self.batch_norm3(self.conv3(output_layer)))
        #print (output_layer.shape)
        output_layer = F.relu(self.batch_norm4(self.conv4(output_layer)))
        #print (output_layer.shape)
        output_layer = F.relu(self.batch_norm5(self.conv5(output_layer)))
        #print (output_layer.shape)
        output_layer = F.relu(self.batch_norm6(self.conv6(output_layer)))
        #print (output_layer.shape)
        output_layer = F.relu(self.batch_norm7(self.conv7(output_layer)))
        #print (output_layer.shape)
        output_layer = F.relu(self.batch_norm8(self.conv8(output_layer)))
        #print (output_layer.shape)
        output_layer = F.relu(self.batch_norm9(self.conv9(output_layer)))
        #print (output_layer.shape)
        output_layer = F.relu(self.batch_norm10(self.conv10(output_layer)))
        #print (output_layer.shape)
        output_layer = F.relu(self.batch_norm11(self.conv11(output_layer)))
        #print (output_layer.shape)
        output_layer = F.relu(self.batch_norm12(self.conv12(output_layer)))
        #print (output_layer.shape)
        output_layer = F.relu(self.batch_norm13(self.conv13(output_layer)))
        output_layer = F.relu(self.batch_norm14(self.conv14(output_layer)))
        output_layer = F.relu(self.batch_norm15(self.conv15(output_layer)))
        
        # output_layer will be (N,8,298,257)
        # we want it to be (N,8*257,298,1)
        batch_size = output_layer.size(0) # N
        height = output_layer.size(2)  # 298
        
        output_layer = output_layer.view(batch_size,-1,height,1)
        return output_layer

class Video_Model(nn.Module):
    
    def __init__(self):
        super(Video_Model, self).__init__()
        
        #video model layers , name of layers as per table 2 given in paper.
        self.linear_for_512_to_1024 = nn.Linear(512,1024)
        
        self.conv1 = nn.Conv2d(1024,256,kernel_size = (7,1),padding = self.get_padding((7,1),(1,1)), dilation = (1,1))
      
        self.conv2 = nn.Conv2d(256,256,kernel_size = (5,1),padding = self.get_padding((5,1),(1,1)), dilation = (1,1))
      
        self.conv3 = nn.Conv2d(256,256,kernel_size = (5,1),padding = self.get_padding((5,1),(2,1)), dilation = (2,1))
      
        self.conv4 = nn.Conv2d(256,256,kernel_size = (5,1),padding = self.get_padding((5,1),(4,1)), dilation = (4,1))
      
        self.conv5 = nn.Conv2d(256,256,kernel_size = (5,1),padding = self.get_padding((5,1),(8,1)), dilation = (8,1))
      
        self.conv6 = nn.Conv2d(256,256,kernel_size = (5,1),padding = self.get_padding((5,1),(16,1)), dilation = (16,1))
      
        #Batch normalization layers
        
        self.batch_norm1 = nn.BatchNorm2d(256)
        self.batch_norm2 = nn.BatchNorm2d(256)
        self.batch_norm3 = nn.BatchNorm2d(256)
        self.batch_norm4 = nn.BatchNorm2d(256)
        self.batch_norm5 = nn.BatchNorm2d(256)
        self.batch_norm6 = nn.BatchNorm2d(256)
        
    def get_padding(self,kernel_size,dilation):
        padding = (((dilation[0])*(kernel_size[0]-1))//2,((dilation[1])*(kernel_size[1]-1))//2)
        return padding
        
    def forward(self,input_video):
        # input video will be (512,75,1)
        if len(input_video.shape) == 3:
            input_video = input_video.unsqueeze(1)
        #input_video = torch.transpose(input_video,1,3) # (1,75,512)
        #print (input_video.shape)
        input_video = self.linear_for_512_to_1024(input_video) # (1,75,1024)
        
        input_video = torch.transpose(input_video,1,3) # (1024,75,1)
        

        output_layer = F.relu(self.batch_norm1(self.conv1(input_video)))
        #print (output_layer.shape)
        output_layer = F.relu(self.batch_norm2(self.conv2(output_layer)))
        #print (output_layer.shape)
        output_layer = F.relu(self.batch_norm3(self.conv3(output_layer)))
        #print (output_layer.shape)
        output_layer = F.relu(self.batch_norm4(self.conv4(output_layer)))
        #print (output_layer.shape)
        output_layer = F.relu(self.batch_norm5(self.conv5(output_layer)))
        #print (output_layer.shape)
        output_layer = F.relu(self.batch_norm6(self.conv6(output_layer)))
        #print (output_layer.shape)
        
        #for upsampling , as mentioned in paper
        output_layer = nn.functional.interpolate(output_layer,size=(298,1),mode="nearest")
        #print (output_layer.shape)
        
        return output_layer

# so now , video_output is (N,256,298,1)
# and audio_output is  (N,8*257,298,1)
# where N = batch_size

class Audio_Visual_Fusion(nn.Module):
    
    def __init__(self,num_person=2,device=None):
        self.device = device
        super(Audio_Visual_Fusion, self).__init__()
        self.num_person = num_person
        self.input_dim = 8*257 + 256*self.num_person
        
        self.audio_output = Audio_Model()
        self.video_output = Video_Model()
        
        self.lstm = nn.LSTM(self.input_dim,400,num_layers=1,bias=True,batch_first=True,bidirectional=True)
        
        self.fc1 = nn.Linear(400,600)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(600,600)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        self.fc3 = nn.Linear(600,600)
        torch.nn.init.xavier_uniform_(self.fc3.weight)

        self.complex_mask_layer = nn.Linear(600,2*257*self.num_person)
        torch.nn.init.xavier_uniform_(self.complex_mask_layer.weight)
    
    def forward(self,input_audio,input_video):
        # input_audio will be (N,2,298,257)
        # input_video will be list of size = num_person , so each item of list will be of (N,512,75,1)
        
        audio_out = self.audio_output(input_audio)
        # audio_out will be (N,256,298,1)
        AVFusion = [audio_out]
        for i in range(self.num_person):
            video_out = self.video_output(input_video[i])
            AVFusion.append(video_out)
        #print (len(AVFusion))
        
        mixed_av = torch.cat(AVFusion,dim=1)
        #print (mixed_av.shape) # will be (N,input_dim,298,1)
        
        mixed_av = mixed_av.squeeze(3)  # (N,input_dim,298)
        mixed_av = torch.transpose(mixed_av,1,2) # (N,298,input_dim)
        
        mixed_av,(h,c) = self.lstm(mixed_av)
        mixed_av = mixed_av[..., :400] + mixed_av[..., 400:]
        
        mixed_av = F.relu(self.fc1(mixed_av))
        mixed_av = F.relu(self.fc2(mixed_av))
        mixed_av = F.relu(self.fc3(mixed_av)) # (N,298,600)
        
        complex_mask = torch.sigmoid(self.complex_mask_layer(mixed_av)) #(N,298,2*257*num_person)
        
        batch_size = complex_mask.size(0) #N
        complex_mask = complex_mask.view(batch_size,2,298,257,self.num_person)
        
        output_audio = torch.zeros(complex_mask.shape).to(self.device)
        for i in range(self.num_person):
            output_audio[..., i] = fast_icRM(input_audio, complex_mask[..., i])
        
        #for i in range(self.num_person):
        #    output_audio[..., i] = complex_mask[..., i] * input_audio 
        
        return output_audio
