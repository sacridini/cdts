# Download de Séries Temporais via Google Earth Engine (GEE)

O `cdts` agora oferece suporte nativo para extração e download de séries temporais usando o **Google Earth Engine (GEE)**. Isso permite que você ignore o download de imagens individuais e deixe o Google processar dados complexos (como harmonização de sensores Landsat, mascaramento de nuvens e composição anual) antes do download.

A nossa infraestrutura não possui dependências pesadas de terceiros (como o geemap) e foi construída focada em performance multithreading e escalabilidade espacial.

## Autenticação

O Earth Engine exige que o usuário autentique sua máquina com as credenciais do Google Cloud Project (GCP) com acesso habilitado à API. Toda vez que você tentar baixar, a função `download_gee_timeseries` cuidará dessa inicialização. Se as suas credenciais expirarem ou ainda não existirem, uma janela do navegador se abrirá solicitando seu login.

> **Nota:** Certifique-se de que a conta Google selecionada tenha acesso ao GEE. É recomendado fornecer o nome do seu projeto GCP usando o parâmetro `project='seu-projeto'`.

## Exemplo 1: Download Direto (Multithread Tiled)

Se você tem uma área pequena a média (ex: município ou um polígono específico) e quer o arquivo `.tif` imediatamente na sua máquina, use o modo `method='direct'`. 

O `cdts` vai fatiar automaticamente sua região em grades menores (*tiles*), abrir dezenas de conexões em paralelo (multithreading) com o Google, baixar os pedaços e juntar tudo (usando `rasterio.merge`) perfeitamente para você.

```python
from cdts.gee import download_gee_timeseries

# Bounding box [min_lon, min_lat, max_lon, max_lat]
meu_roi = [-47.95, -15.85, -47.85, -15.75]

download_gee_timeseries(
    roi=meu_roi, 
    start_date='2010-01-01',
    end_date='2020-12-31', 
    out_dir='./gee_dados_diretos',
    method='direct',           # Habilita o download e mosaico local na hora
    composite_type='annual',   # Gera Mosaico Medoid Anual no estilo LandTrendr
    project='meu-projeto-gcp'  # Substitua pelo ID do seu projeto no Google Cloud
)
```

## Exemplo 2: Exportando para o Google Drive (Grandes Áreas)

Para análises em escala estadual ou nacional, baixar os dados diretamente pela internet na mesma hora pode falhar devido aos limites de payload da API ou levar muito tempo. 

Nesses casos, passe o `method='drive'`. O `cdts` configurará tudo e enviará uma tarefa (Task) diretamente para os servidores do Google. O Google salvará o arquivo final silenciosamente na nuvem do seu **Google Drive**, na pasta `CDTS_Downloads`.

```python
from cdts.gee import download_gee_timeseries

estado_sp = [-53.11, -25.31, -44.15, -19.78]

download_gee_timeseries(
    roi=estado_sp, 
    start_date='1985-01-01',
    end_date='2022-12-31', 
    out_dir='./dados', # Usado apenas para nomear os arquivos no Drive neste modo
    method='drive',    # Inicia a exportação assíncrona
    composite_type='annual',
    project='meu-projeto-gcp'
)

# O terminal vai exibir uma mensagem semelhante a:
# [landsat_medoid_1985] Task sent to Google Drive (Task ID: ABCD123456).
```

## O que está acontecendo por baixo dos panos?

Ao usar o `composite_type='annual'` (padrão atual do `cdts` para integrações com o LandTrendr):

1. **Fusão de Sensores:** A função busca coleções do Landsat 5, 7, 8 e 9 (*Surface Reflectance Collection 2*).
2. **Harmonização:** Os valores do Landsat 8 e 9 (OLI) são convertidos matematicamente para os equivalentes ETM+ usando os coeficientes do artigo científico de Roy et al. (2016), garantindo uma série temporal perfeita, livre de vieses de sensores.
3. **Máscara de Nuvens:** A banda de garantia de qualidade `QA_PIXEL` é usada para filtrar nuvens densas e sombras em todas as imagens.
4. **Composição Medoid:** Em vez da simples mediana, aplicamos a estratégia geométrica Medoid para achar o pixel real que mais representa a estação (normalmente usado em fluxos de trabalho *eMapR/LandTrendr*).
