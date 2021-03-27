import SimpleITK as sitk 
import matplotlib.pyplot as plt 
import os 
import pandas as pd
import numpy as np

'''
Transforma Array em Image
'''
def A2I (Array):
    return sitk.GetImageFromArray(Array)

'''
Transforma Image em Array
'''
def I2A (Image):
    return sitk.GetArrayFromImage(Image)

'''
Lê uma pasta com arquivos DICOM no formato .ima ou .dcm
'''
def lePastaDICOM (path_dicom):
    
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(path_dicom) 
    reader.SetFileNames(dicom_names) 
    
    image = reader.Execute() 
    
    print(f'Size: {image.GetSize()}; Spacing: {image.GetSpacing()}')
    
    return image

'''
Mostra cortes de imagens DICOM
'''
def mostraCortesDICOM(imagensDICOM,idx_inicial,n_cortes,colorbar=False):
    imagens = sitk.GetArrayFromImage(imagensDICOM)
    fig = mostraCortes(imagens,idx_inicial,n_cortes,colorbar=colorbar)
    return fig

'''
Mostra cortes de um array de imagens
'''
def mostraCortes(imagens,idx_inicial,n_cortes,colorbar=False):
    fig = plt.figure(figsize=(20,5*n_cortes/2))
    for idx in range(n_cortes):
        plt.subplot(int(n_cortes/2)+1,2,idx+1)
        plt.imshow(imagens[idx+idx_inicial])
        plt.title(f'slice {idx+idx_inicial}')
        if colorbar:
            plt.colorbar()
    return fig

'''
Esta funcao segmenta o pulmao e retorna um DataFrame com os seguintes dados:
 - id: enumera as imagens
 - imagem: armazena as imagens em formato numpy array
 - mascara: armazena a mascara de segmentacao em formato numpy array
 - imagem segmentada: armazena a imagem segmentada em formato numpy array
'''
def SegmentaPulmaoAerado(imagem,lower=-2048,upper=-200,maxvalue=-100):
    biblioteca = {'id':[], 'imagem':[], 'mascara_aerado':[], 'imagem_aerado_segmentada':[]}
    count = 0
    for i in sitk.GetArrayFromImage(imagem):
        #fundo
        lstSeeds = [(0,0),(511,0),(511,511),(0,511)]
        imagem_fundo = sitk.ConnectedThreshold(image1 = imagem[:,:,count], seedList=lstSeeds, lower=lower, upper=upper)
        imagem_fundo_array = sitk.GetArrayFromImage(imagem_fundo)
        seg_array = sitk.GetArrayFromImage(imagem[:,:,count]) < maxvalue
        pulmao_array = seg_array ^ imagem_fundo_array
        
        #abrindo
        raio = (4,4)
        pulmao_erode = sitk.BinaryErode(sitk.GetImageFromArray(pulmao_array*1), raio)
        pulmao_dilate = sitk.BinaryDilate(pulmao_erode, raio)
    
        #fechando
        pulmao_dilate1 = sitk.BinaryDilate(pulmao_dilate, raio)
        pulmao_erode1 = sitk.BinaryErode(pulmao_dilate1, raio)
        
       #mascara
        masc_lung_seg = sitk.GetArrayFromImage(pulmao_erode1) == 1
        
        #imagem do pulmão segmentado
        count+=1
        biblioteca['id'].append(count)
        count-=1
        imagem_pulmao_seg = sitk.GetArrayFromImage(imagem[:,:,count]) * masc_lung_seg
        biblioteca['imagem'].append(sitk.GetArrayFromImage(imagem[:,:,count]))
        biblioteca['mascara_aerado'].append(masc_lung_seg)
        biblioteca['imagem_aerado_segmentada'].append(imagem_pulmao_seg)
        count+=1
        df = pd.DataFrame(biblioteca)
        
    return df

'''
Este método segmenta o pulmão de suínos, incluindo a parte colapsada,
considerando que a região anterio do pulmão está aerada e a região
posterior está colapsada.
'''
def SegmentaPulmaoCompleto(imagens, threshold = 200, debug = False, 
                             raio_abertura = (3,3), raio_fechamento = (60,60), raio_dilat = (5,5),
                             altura_limite_aerado = -1):
    biblioteca = {'imagem':[], 'mascara_ar':[], 'imagem_ar':[]}
    nl,nc,nimagens = imagens.GetSize()
    print(f'Tamanho: {nl} {nc} {nimagens}')
    for idx in range(nimagens):
        imgct_Image = imagens[:,:,idx]
        imgct = sitk.GetImageFromArray(sitk.GetArrayFromImage(imgct_Image)) # VERIFICAR PORQUE EH NECESSARIO...
        mascara_pulmao_Image = SegmentaPulmaoCompletoImg(imgct,threshold, debug, raio_abertura, raio_fechamento, raio_dilat,    altura_limite_aerado)
        mascara_pulmao = sitk.GetArrayFromImage(mascara_pulmao_Image)
        imagem = sitk.GetArrayFromImage(imagens[:,:,idx])
        imagem_pulmao = (mascara_pulmao!=0)*imagem
        biblioteca['imagem'].append(imagem)
        biblioteca['mascara_ar'].append(mascara_pulmao)
        biblioteca['imagem_ar'].append(imagem_pulmao)
    df = pd.DataFrame(biblioteca)
    
  
    return df


'''
Separa um tecido dos demais no mesmo label, por conectividade
'''
def SeparaTecidoConectado(mascaras,label,altura=0.5):
    mascaras_Image = A2I(mascaras.tolist())
    nc,nl,nimagens = mascaras_Image.GetSize()
    
    # Encontra semente. A semente é o primeiro pixel do valor de 'label' em um slice a uma certa altura:
    sliceCentro = int(nimagens*altura)
    imagem_centro = I2A(mascaras_Image[:,:,sliceCentro])
    result = np.where(imagem_centro == label)
    seed_lst = [(int(result[1][0]), int(result[0][0]), int(sliceCentro))] # cuidado! indices devem estar invertidos
    mascara_tecido_Image = sitk.ConnectedThreshold(mascaras_Image,
                                                   seedList=seed_lst,
                                                   lower=label,
                                                   upper=label,
                                                   replaceValue=1,
                                                   connectivity=0)
    return mascara_tecido_Image
    

def SegmentaTraqueia(imagens_ct, mascara_pulmao, threshold = -900,altura=0.9,label = 1):
    imagens_pulmao = I2A(imagens_ct)*I2A(mascara_pulmao)
    
    mask_traqueia = imagens_pulmao<threshold
    mask_traqueia_img = sitk.GetImageFromArray(mask_traqueia*1)
    mask_traqueia1 = sitk.BinaryErode(mask_traqueia_img, (2,2,0))
    
    nl,nc,nimagens = imagens_ct.GetSize()
    seed_lst = []
    inicio = int(np.floor(nimagens*altura))
    print((inicio,nimagens))
    for idx in range(inicio,nimagens): # pega uma semente em cada slice
        imagem_slice = I2A(mask_traqueia1[:,:,idx])
        result = np.where(imagem_slice == label)
        seed_lst.append((int(result[1][0]), int(result[0][0]), int(idx)))
    mascara_tecido_Image = sitk.ConnectedThreshold(mask_traqueia1,
                                                   seedList=seed_lst,
                                                   lower=label,
                                                   upper=label,
                                                   replaceValue=1,
                                                   connectivity=0)
    mask_traqueia2 = sitk.BinaryDilate(mascara_tecido_Image, (2,2,1))

    mask_traqueia3 = sitk.BinaryDilate(mask_traqueia2, (5,5,10))
    mask_traqueia4 = sitk.BinaryErode(mask_traqueia3, (5,5,10))

    
    return mask_traqueia4
    
    
    
'''
Este método segmenta o pulmão de suínos (apenas uma imagem), incluindo a parte colapsada,
considerando que a região anterio do pulmão está aerada e a região
posterior está colapsada.
'''
def SegmentaPulmaoCompletoImg(imagem, threshold = 200, debug = False, 
                             raio_abertura = (3,3), raio_fechamento = (60,60), raio_dilat = (5,5),
                             altura_limite_aerado = -1):

    # Encontrando fundo da imagem:
    tamanho_x,tamanho_y = imagem.GetSize()
    lstSeeds = [(0,0),(tamanho_x-1,0),(tamanho_x-1,tamanho_y-1),(0,tamanho_y-1)] # 'cantos' da imagem
    
    # inclui toda a lateral da imagem como sementes para incluir colchão:
    for idy in range(5,tamanho_y,5):
        lstSeeds.append((0,idy))
        lstSeeds.append((tamanho_x-1,idy))
        
    imagem_fundo = sitk.ConnectedThreshold(image1=imagem, seedList=lstSeeds, lower=-1100, upper=-200)
    
    # Dilata o fundo da imagem para incluir a espessura da caixa torácica:    
    imagem_fundo_dilatada = sitk.BinaryDilate(imagem_fundo, (50,50))
    
    # inclui os ossos na imagem obtida
    imagem_fundo_com_ossos = (imagem > threshold) + imagem_fundo_dilatada
    
    # impõe um fechamento com raio grande para juntar costelas e fundo
    imagem_fechamento_grande_temp = sitk.BinaryDilate(imagem_fundo_com_ossos, raio_fechamento)
    imagem_fechamento_grande = sitk.BinaryErode(imagem_fechamento_grande_temp, raio_fechamento)
    
    # dilatando um pouco o resultado para evitar sobreposição das costelas
    imagem_fechamento_grande_dilatado = sitk.BinaryDilate(imagem_fechamento_grande, raio_dilat)
    
    # encontrando a parte aerada do pulmão para usar na parte anterior da segmentação
    pulmao,_ = encontra_pulmao(imagem)
    pulmao_array = sitk.GetArrayFromImage(pulmao)

    # Aplica máscara da parte aerada do pulmão ao que foi encontrado anteriormente
    imagem_considerando_aerado = imagem_fechamento_grande_dilatado & (1-pulmao)
    imagem_considerando_aerado_array = sitk.GetArrayFromImage(imagem_considerando_aerado)

    # Se altura_limite_aerado == -1, encontra altura ideal
    if altura_limite_aerado == -1:
        pulmao_projection = pulmao_array.sum(axis=1)
        altura_limite_aerado = np.argmax(pulmao_projection)
    
    # Substitui regiao anterior (aerada) pela segmentação mais confiável
    imagem_considerando_aerado_array[0:altura_limite_aerado,:] = 1-pulmao_array[0:altura_limite_aerado,:]
    imagem_considerando_aerado_novo = sitk.GetImageFromArray(imagem_considerando_aerado_array)
    
    # Aplica uma abertura para remover qualquer ruído deixado pela parte aerada
    imagem_abertura_temp = sitk.BinaryErode(imagem_considerando_aerado_novo, raio_abertura)
    imagem_abertura = sitk.BinaryDilate(imagem_abertura_temp, raio_abertura)
    
    # imprime informações de debug
    if debug:
        print('Altura_limite_aerado utilizada: {}'.format(altura_limite_aerado))
        debuga_imagens((imagem_fundo,imagem_fundo_com_ossos,imagem_fechamento_grande_dilatado,
                    pulmao,imagem_considerando_aerado,imagem_abertura))
        
    # pulmão completo
    mask_pulmao_completo = 1-imagem_abertura
    
    # separa parte aerada
    mask_pulmao_completo_a = I2A(mask_pulmao_completo)
    mask_pulmao_completo_a[I2A(pulmao)==1] = 2; # label pulmao_aerado: 2
    
    
    # retorna a máscara do pulmão
    return A2I(mask_pulmao_completo_a)

def encontra_pulmao(imagem, threshold = -100, debug = False, raio_abertura = (4,4), raio_fechamento = (4,4)):
    # Encontrando fundo da imagem:
    tamanho_x,tamanho_y = imagem.GetSize()
    lstSeeds = [(0,0),(tamanho_x-1,0),(tamanho_x-1,tamanho_y-1),(0,tamanho_y-1)] # 'cantos' da imagem
    
    # inclui toda a lateral da imagem como sementes para incluir colchão:
    for idy in range(5,tamanho_y,5):
        lstSeeds.append((0,idy))
        lstSeeds.append((tamanho_x-1,idy))
    
    imagem_fundo = sitk.ConnectedThreshold(image1=imagem, seedList=lstSeeds, lower=-1100, upper=-200)
    
    imagem2 = imagem < threshold # verificando pixels abaixo do threshold
    imagem_pulmao = imagem2 ^ imagem_fundo # removendo fundo
    imagem_pulmao2 = sitk.BinaryErode(imagem_pulmao, raio_abertura) # abertura passo 1
    imagem_pulmao3 = sitk.BinaryDilate(imagem_pulmao2, raio_abertura) # abertura passo 2
    imagem_pulmao4 = sitk.BinaryDilate(imagem_pulmao3, raio_fechamento) # fechamento passo 1
    imagem_pulmao5 = sitk.BinaryErode(imagem_pulmao4, raio_fechamento) # fechamento passo 2
    imagem_pulmao_seg =  sitk.GetImageFromArray(sitk.GetArrayFromImage(imagem) *
                                                sitk.GetArrayFromImage(imagem_pulmao5) ) # aplica máscara
    
    # Mostrando resultado:
    if debug:
        print("Imagem {} x {} pixels.".format(tamanho_x,tamanho_y))
        imagens = (sitk.GetArrayFromImage(imagem2),
                   sitk.GetArrayFromImage(imagem_fundo),
                   sitk.GetArrayFromImage(imagem_pulmao),
                   sitk.GetArrayFromImage(imagem_pulmao3),
                   sitk.GetArrayFromImage(imagem_pulmao5),
                   sitk.GetArrayFromImage(imagem_pulmao_seg)-1000*(1-sitk.GetArrayFromImage(imagem_pulmao5)),
                   sitk.GetArrayFromImage(imagem))
        titulos = ('Depois do threshold',
                   'Fundo',
                   'Removendo fundo',
                   'Depois da abertura',
                   'Depois do fechamento',
                   'Pulmao segmentado',
                   'Imagem original')

        fig, axs = plt.subplots(1,len(titulos),figsize=(20,5))
        for ax,img,titulo in zip(axs,imagens,titulos):
            ax.imshow(img)
            ax.set_title(titulo)
            ax.axis('off')
    
    return imagem_pulmao5, imagem_pulmao_seg

