# digits-28

Uma rede convolucional treinada no MNIST que reconhece dígitos manuscritos **inteiramente dentro do navegador**. Depois que a página carrega, não existe servidor, requisição nem internet no caminho: o modelo roda no dispositivo de quem abriu.

**Página publicada:** https://hugbrl09.github.io/digits-28/

---

## Passo a passo para colocar no ar

O repositório já traz a página pronta. Falta apenas o modelo, que é treinado no Colab.

### 1. Treinar no Google Colab

Abra `treinar_mnist_colab.ipynb` no Colab (`Arquivo → Fazer upload de notebook`, ou pelo botão *Open in Colab* se o repositório for público).

Antes de rodar, ative a GPU: **Ambiente de execução → Alterar o tipo de ambiente de execução → T4 GPU**.

Depois, `Ambiente de execução → Executar tudo`. O treino leva de 3 a 6 minutos na T4. A última célula baixa um **`entrega.zip`**.

### 2. Colocar os arquivos no repositório

Descompacte o `entrega.zip` na raiz do projeto. Ele contém:

| Arquivo | Para quê |
|---|---|
| `modelo_web/model.json` + `.bin` | o modelo que o navegador carrega |
| `relatorio.json` | os números que a página exibe na seção de avaliação |
| `modelo.keras` | o modelo original, para retreinar sem repetir o treino |

### 3. Testar localmente

O navegador bloqueia `fetch` em `file://`, então é preciso um servidor estático:

```bash
python -m http.server 8000
# abra http://localhost:8000
```

Se a previsão funcionar e a seção **Avaliação** aparecer preenchida, está pronto.

### 4. Publicar

```bash
git add -A
git commit -m "feat: modelo treinado e relatorio de avaliacao"
git push
```

Depois, no GitHub: **Settings → Pages → Source: Deploy from a branch**, escolha a branch e a pasta `/ (root)`, e **Save**. O link fica disponível em cerca de um minuto.

> Enquanto o modelo não estiver publicado, a página carrega normalmente e mostra um aviso explicando o que falta — nada quebra.

---

## Parte 1 · Treinar

Treino sobre o MNIST completo: 60.000 imagens de treino e 10.000 de teste.

```
Entrada 28 × 28 × 1
│
├─ Bloco 1   Conv 32 (3×3) → Conv 32 (3×3) → BatchNorm → MaxPool → Dropout 0,25   28×28 → 14×14
├─ Bloco 2   Conv 64 (3×3) → Conv 64 (3×3) → BatchNorm → MaxPool → Dropout 0,25   14×14 →  7×7
├─ Bloco 3   Conv 128 (3×3)                → BatchNorm → MaxPool → Dropout 0,25    7×7  →  3×3
│
└─ Flatten (1152) → Dense 128 → BatchNorm → Dropout 0,4 → Dense 10 (softmax)
```

**289.130 parâmetros · 1,1 MB exportados.**

| Requisito do enunciado | Onde está |
|---|---|
| entrada 28 × 28 × 1 | `keras.Input(shape=(28, 28, 1))` |
| pelo menos duas camadas convolucionais | cinco: `conv1a`, `conv1b`, `conv2a`, `conv2b`, `conv3a` |
| saída `Dense(10, activation='softmax')` | camada `saida` |
| perda `sparse_categorical_crossentropy` | no `compile()` |
| `EarlyStopping` com `restore_best_weights=True` | `patience=6`, monitorando `val_loss` |

### Justificativa da arquitetura

**Duas convoluções antes de cada pooling.** Duas 3×3 empilhadas cobrem a mesma região que uma 5×5, com menos parâmetros e uma não-linearidade a mais no meio. Traço manuscrito é feito de bordas e curvas curtas — profundidade barata rende mais do que filtro largo.

**Filtros dobrando a cada bloco.** Cada pooling corta a resolução pela metade. Dobrar os canais compensa a perda espacial, deslocando a representação de *onde* está a borda para *que tipo* de traço ela é.

**BatchNormalization.** Normalizar as ativações dentro do bloco estabiliza o gradiente e permite uma taxa de aprendizado maior sem divergir. É o que faz a rede chegar perto de 99% em poucas épocas em vez de dezenas.

**Dropout crescente (0,25 → 0,4).** As convoluções já são regularizadas pelo compartilhamento de pesos; a densa de 128 unidades é a que mais decora, então o dropout mais forte fica onde está o risco.

**Mapa de 3×3×128 antes do `Flatten`.** Chegar a um mapa espacial pequeno antes de achatar mantém a camada densa em 147 mil parâmetros em vez de 400 mil e sobrando. O modelo inteiro fica em ~1,1 MB — e esse arquivo é baixado pelo navegador de quem abre a página, então tamanho aqui é experiência de uso, não elegância.

### Duas decisões que não estavam no enunciado

**A validação sai do treino, não do teste.** O `EarlyStopping` precisa de um conjunto para decidir quando parar. Usar os 10.000 exemplos de teste para isso vazaria informação e inflaria a acurácia reportada. O notebook separa 6.000 imagens do próprio treino para validação e só toca no teste na Parte 2.

**Aumento de dados, aplicado fora do modelo.** Rotação ±14°, translação ±10% e zoom ±10%. O MNIST é composto de dígitos já centralizados e normalizados; o usuário desenha torto, deslocado e com espessura variável. O aumento aproxima o treino da distribuição real de uso. As camadas de aumento vivem no `tf.data`, não dentro da rede — assim o modelo exportado carrega só o que a inferência precisa.

---

## Parte 2 · Avaliar antes de exportar

O notebook calcula, sobre os 10.000 exemplos de teste:

- acurácia e perda;
- **matriz de confusão 10 × 10**, com precisão, revocação e F1 por classe;
- os **dois dígitos mais confundidos entre si** — somando os dois sentidos (`4→9` mais `9→4`), que é o que a palavra "entre si" pede; a lista direcional também é reportada;
- as **imagens que ele errou**, ordenadas pelos erros mais confiantes: os casos em que o modelo errou *e ainda assim* tinha certeza alta são os que revelam atalhos ruins aprendidos pela rede.

Tudo isso é gravado em `relatorio.json`, e **a página lê esse arquivo** para montar a seção *Avaliação*. Nenhum número está escrito à mão no HTML: retreinar o modelo atualiza o relatório publicado junto. Não existe a possibilidade de a página exibir a acurácia de um modelo diferente do que está carregado nela.

---

## Parte 3 · Exportar

O notebook tenta duas rotas, nesta ordem:

1. **`tensorflowjs` oficial** — `tfjs.converters.save_keras_model(...)`, o caminho do enunciado.
2. **Exportador embutido** — o pacote `tensorflowjs` costuma exigir uma versão de TensorFlow diferente da que o Colab traz, e a instalação quebra o runtime com frequência. A função `exportar_tfjs` escreve o mesmo formato `layers-model` direto do Keras 3, sem dependência extra: sanitiza os campos que o Keras 3 emite e o parser do TF.js não conhece (`batch_shape` → `batch_input_shape`, `dtype` como dicionário de política → string) e grava os pesos em float32 nomeados como `<camada>/<peso>`.

A célula seguinte **confere a exportação**: compara nome e forma de cada tensor de peso do `model.json` com o modelo em memória e valida o tamanho do `.bin` em bytes. Se algo divergir, ela falha em vez de deixar publicar um modelo quebrado.

O exportador foi verificado carregando o `model.json` gerado no próprio TensorFlow.js e comparando a saída com a do Keras sobre a mesma entrada: **diferença máxima de 1,5 × 10⁻⁸**.

---

## Parte 4 · A página

`index.html` — um arquivo, sem framework e sem build. As únicas dependências externas são o TensorFlow.js por CDN e as fontes do Google, ambas carregadas junto com a página.

- canvas com desenho a **mouse, dedo e caneta** (Pointer Events, um só caminho de código);
- botão **limpar**, além de desfazer, borracha e três espessuras;
- **dígito previsto em destaque**, com anel de confiança;
- **as 10 probabilidades em barras**, com o primeiro e o segundo colocados diferenciados;
- previsão **ao vivo**, enquanto o traço acontece;
- atalhos de teclado — `0`–`9` desenham um exemplo, `C` limpa, `E`/`P` trocam a ferramenta, `Ctrl+Z` desfaz;
- tema claro e escuro;
- **o que o modelo vê**: a grade 28×28 depois do pré-processamento, ao vivo;
- **mapas de ativação** dos 32 filtros da primeira convolução reagindo ao desenho.

### O pré-processamento (a parte que faz funcionar)

O MNIST não é "o dígito como a pessoa desenhou". Cada imagem foi recortada, reescalada para caber numa caixa de 20×20 e recolocada numa grade 28×28 de modo que o **centro de massa** dos pixels caia no meio. Um modelo treinado nisso espera exatamente isso — mandar a tela inteira, com o dígito no canto, é a origem mais comum de erro nesse tipo de projeto.

A página repete o mesmo procedimento antes de cada previsão:

1. rasteriza o traço numa tela auxiliar 280×280, preta com tinta branca;
2. acha a caixa delimitadora dos pixels com tinta;
3. reduz o recorte para caber em 20×20 preservando a proporção;
4. calcula o centro de massa e cola o recorte na grade 28×28 deslocado para que esse ponto caia no centro;
5. divide por 255 e monta o tensor `[1, 28, 28, 1]`.

Dois detalhes que mudam o resultado:

**Duas telas, não uma.** A tela visível tem traço colorido com brilho; a tela que o modelo lê é preta e branca, sem efeito nenhum. Separar as duas permite estilizar o desenho à vontade sem nunca sujar a entrada da rede.

**Média por área na redução, não `drawImage`.** Numa redução de ~14×, o reamostrador do navegador descarta pixels e o traço chega picotado à rede, o que muda bastante a previsão. O passo 3 faz a média dos pixels de cada célula de destino — o mesmo que o MNIST fez ao construir o dataset.

---

## Estrutura

```
digits-28/
├── index.html                  a página inteira: markup, estilos e lógica
├── favicon.svg
├── treinar_mnist_colab.ipynb   treino, avaliação e exportação (Partes 1 a 3)
├── modelo_web/                 vem do notebook
│   ├── model.json
│   └── group1-shard1of1.bin
├── relatorio.json              vem do notebook — alimenta a seção Avaliação
├── modelo.keras                vem do notebook
└── .nojekyll                   impede o Jekyll do GitHub Pages de mexer nos arquivos
```

---

## Retreinar

Rode o notebook de novo, troque `modelo_web/` e `relatorio.json` pelos novos arquivos e faça o push. A página se ajusta sozinha: a tabela de arquitetura, os parâmetros, a matriz de confusão e as amostras de erro vêm todos do `relatorio.json`.
