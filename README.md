# digits-28

Uma rede convolucional treinada no MNIST que reconhece dígitos manuscritos **dentro do navegador**. Depois que a página carrega, não existe servidor, requisição nem internet no caminho: o modelo roda no dispositivo de quem abriu.

**Página:** https://hugbrl09.github.io/digits-28/

**99,59%** de acurácia no conjunto de teste · **289.130** parâmetros · **1,1 MB** baixados uma única vez.

---

## Parte 1 · Treinar

MNIST completo: 60.000 imagens de treino e 10.000 de teste.

```
Entrada 28 × 28 × 1
│
├─ Bloco 1   Conv 32 (3×3) → Conv 32 (3×3) → BatchNorm → MaxPool → Dropout 0,25   28×28 → 14×14
├─ Bloco 2   Conv 64 (3×3) → Conv 64 (3×3) → BatchNorm → MaxPool → Dropout 0,25   14×14 →  7×7
├─ Bloco 3   Conv 128 (3×3)                → BatchNorm → MaxPool → Dropout 0,25    7×7  →  3×3
│
└─ Flatten (1152) → Dense 128 → BatchNorm → Dropout 0,4 → Dense 10 (softmax)
```

| Requisito | Onde está |
|---|---|
| entrada 28 × 28 × 1 | `keras.Input(shape=(28, 28, 1))` |
| pelo menos duas camadas convolucionais | cinco: `conv1a`, `conv1b`, `conv2a`, `conv2b`, `conv3a` |
| saída `Dense(10, activation='softmax')` | camada `saida` |
| perda `sparse_categorical_crossentropy` | no `compile()` |
| `EarlyStopping` com `restore_best_weights=True` | `patience=6`, monitorando `val_loss` |

### Justificativa da arquitetura

**Duas convoluções antes de cada pooling.** Duas 3×3 empilhadas cobrem a mesma região que uma 5×5, com menos parâmetros e uma não-linearidade a mais no meio. Traço manuscrito é feito de bordas e curvas curtas — profundidade barata rende mais do que filtro largo.

**Filtros dobrando a cada bloco.** Cada pooling corta a resolução pela metade. Dobrar os canais compensa a perda espacial, deslocando a representação de *onde* está a borda para *que tipo* de traço ela é.

**BatchNormalization.** Normalizar as ativações dentro do bloco estabiliza o gradiente e permite uma taxa de aprendizado maior sem divergir. É o que faz a rede passar de 88% para mais de 99% em poucas épocas.

**Dropout crescente (0,25 → 0,4).** As convoluções já são regularizadas pelo compartilhamento de pesos; a densa de 128 unidades é a que mais decora, então o dropout mais forte fica onde está o risco.

**Mapa de 3×3×128 antes do `Flatten`.** Chegar a um mapa espacial pequeno antes de achatar mantém a camada densa em 147 mil parâmetros em vez de 400 mil. O modelo inteiro fica em 1,1 MB — e esse arquivo é baixado por quem abre a página, então tamanho aqui é experiência de uso.

### Duas decisões que o enunciado não pedia

**A validação sai do treino, não do teste.** O `EarlyStopping` precisa de um conjunto para decidir quando parar. Usar os 10.000 exemplos de teste para isso vazaria informação e inflaria a acurácia reportada. O notebook separa 6.000 imagens do próprio treino para validação e só toca no teste na Parte 2.

**Aumento de dados, aplicado fora do modelo.** Rotação ±14°, translação ±10% e zoom ±10%. O MNIST tem dígitos já centralizados e normalizados; quem usa a página desenha torto, deslocado e com espessura variável. As camadas de aumento vivem no `tf.data`, não dentro da rede — assim o modelo exportado carrega só o que a inferência precisa.

---

## Parte 2 · Avaliar antes de exportar

Medido sobre os 10.000 exemplos de teste, que não foram usados em nenhum momento do treino.

| | |
|---|---|
| **Acurácia no teste** | **99,59%** |
| Perda no teste | 0,0104 |
| Erros | 41 de 10000 |
| Épocas executadas | 36 (melhor: 30) |
| Treinado em | Tesla T4, TensorFlow 2.20.0 |

### Matriz de confusão 10 × 10

Linha = rótulo verdadeiro, coluna = previsão do modelo. A diagonal em negrito são os acertos; `·` é ausência de erro.

| real \ previsto | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| **0** | **979** | · | · | 1 | · | · | · | · | · | · |
| **1** | · | **1131** | · | 1 | · | · | 1 | 2 | · | · |
| **2** | · | · | **1027** | 1 | · | · | · | 3 | 1 | · |
| **3** | · | · | · | **1009** | · | 1 | · | · | · | · |
| **4** | · | 1 | · | · | **979** | · | · | · | · | 2 |
| **5** | · | · | · | 3 | · | **887** | 2 | · | · | · |
| **6** | 1 | 3 | · | · | 1 | 1 | **949** | · | 3 | · |
| **7** | · | 1 | 2 | · | · | · | · | **1025** | · | · |
| **8** | 1 | 1 | · | 1 | · | 1 | · | · | **970** | · |
| **9** | · | · | · | · | 5 | · | · | · | 1 | **1003** |

### Os dois dígitos que o modelo mais confunde entre si

**4 ↔ 9**, com **7 erros** somando os dois sentidos: 2 imagens do 4 lidas como 9, e 5 do 9 lidas como 4.

É o par clássico do MNIST — manuscrito, a distância entre um 4 de topo fechado e um 9 de perna reta é de poucos pixels. E é exatamente o tipo de coisa que a acurácia sozinha esconde: 99,59% não diz *onde* o modelo erra, a matriz diz.

Os cinco pares mais confundidos:

| par | erros somando os dois sentidos |
|---|---|
| 4 ↔ 9 | 7 |
| 2 ↔ 7 | 5 |
| 1 ↔ 6 | 4 |
| 3 ↔ 5 | 4 |
| 1 ↔ 7 | 3 |

### Três imagens que ele errou

<img src="relatorio/erro-1.png" width="96" alt="dígito 4 classificado como 9" />
<img src="relatorio/erro-2.png" width="96" alt="dígito 5 classificado como 6" />
<img src="relatorio/erro-3.png" width="96" alt="dígito 2 classificado como 7" />

**1.** previu **9**, era **4** (99,9% de certeza) · **2.** previu **6**, era **5** (99,6% de certeza) · **3.** previu **7**, era **2** (99,1% de certeza)

São os erros mais confiantes do conjunto: casos em que o modelo errou *e ainda assim* atribuiu alta probabilidade à classe errada. As outras 5 amostras estão no `relatorio.json`.

### Desempenho por classe

| dígito | precisão | revocação | F1 | suporte |
|---|---|---|---|---|
| 0 | 99,80% | 99,90% | 99,85% | 980 |
| 1 | 99,47% | 99,65% | 99,56% | 1135 |
| 2 | 99,81% | 99,52% | 99,66% | 1032 |
| 3 | 99,31% | 99,90% | 99,61% | 1010 |
| 4 | 99,39% | 99,69% | 99,54% | 982 |
| 5 | 99,66% | 99,44% | 99,55% | 892 |
| 6 | 99,68% | 99,06% | 99,37% | 958 |
| 7 | 99,51% | 99,71% | 99,61% | 1028 |
| 8 | 99,49% | 99,59% | 99,54% | 974 |
| 9 | 99,80% | 99,41% | 99,60% | 1009 |

O dígito 6 é o de menor revocação (99,06%): três imagens dele foram lidas como 1 e três como 8.

### Curva de aprendizado

O `EarlyStopping` parou na época 36 e restaurou os pesos da época 30.

| | primeira época | melhor época (30) |
|---|---|---|
| Acurácia no treino | 88,45% | 99,41% |
| Acurácia na validação | 88,40% | 99,53% |
| Perda na validação | 0,4378 | 0,0150 |

> **Nota.** Recalculando a mesma avaliação em CPU, dá 99,60% (40 erros) em vez de 99,59% (41). A diferença é uma única imagem de um `6` cuja margem entre as duas classes candidatas é exatamente `0,00e+00` — a menor de todo o conjunto. A T4 desempata para `8`, a CPU para `6`. É diferença de arredondamento entre kernels, não divergência de pesos. Os números acima são os da execução no Colab.

---

## Parte 3 · Exportar

O notebook tenta duas rotas, nesta ordem:

1. **`tensorflowjs` oficial** — `tfjs.converters.save_keras_model(...)`, o caminho do enunciado.
2. **Exportador embutido** — o pacote `tensorflowjs` costuma exigir uma versão de TensorFlow diferente da que o Colab traz, e a instalação quebra o runtime com frequência. A função `exportar_tfjs` escreve o mesmo formato `layers-model` direto do Keras 3, sem dependência extra: sanitiza os campos que o Keras 3 emite e o parser do TF.js não conhece (`batch_shape` → `batch_input_shape`, `dtype` como dicionário de política → string) e grava os pesos em float32 nomeados como `<camada>/<peso>`.

No treino que gerou este modelo, a rota oficial falhou e o exportador embutido assumiu.

A célula seguinte **confere a exportação**: compara nome e forma de cada tensor de peso do `model.json` com o modelo em memória e valida o tamanho do `.bin` em bytes. Se algo divergir, ela falha em vez de deixar publicar um modelo quebrado.

Verificação adicional feita fora do notebook: o `modelo_web/` publicado foi carregado no próprio TensorFlow.js e comparado ao `modelo.keras` sobre imagens reais de teste — **diferença máxima de 3,5 × 10⁻¹⁰**.

---

## Parte 4 · A página

`index.html` — um arquivo, sem framework e sem build. As únicas dependências externas são o TensorFlow.js por CDN e as fontes do Google, ambas carregadas junto com a página.

O que existe na página:

- canvas onde se desenha com **mouse e com o dedo** (Pointer Events cobrem mouse, toque e caneta pelo mesmo caminho de código);
- **botão para limpar**;
- **dígito previsto em destaque**;
- **a confiança de cada uma das 10 classes, em barras**.

A previsão acontece ao vivo, enquanto o traço é feito. As teclas `0`–`9` desenham um dígito de exemplo, útil para testar sem desenhar.

### O pré-processamento

O MNIST não guarda "o dígito como a pessoa desenhou". Cada imagem foi recortada, reescalada para caber numa caixa de 20×20 e recolocada numa grade 28×28 de modo que o **centro de massa** dos pixels caia no meio. Um modelo treinado nisso espera exatamente isso — mandar a tela inteira, com o dígito no canto, é a origem mais comum de erro nesse tipo de projeto.

A página repete o mesmo procedimento antes de cada previsão:

1. rasteriza o traço numa tela auxiliar 280×280, preta com tinta branca;
2. acha a caixa delimitadora dos pixels com tinta;
3. reduz o recorte para caber em 20×20 preservando a proporção;
4. calcula o centro de massa e cola o recorte na grade 28×28 deslocado para que esse ponto caia no centro;
5. divide por 255 e monta o tensor `[1, 28, 28, 1]`.

A miniatura ao lado do botão *Limpar* mostra o resultado desse processo — é literalmente o que entra na rede.

Dois detalhes que mudam o resultado:

**Duas telas, não uma.** A tela visível é tinta escura sobre papel, porque é o que parece natural desenhar. A tela que o modelo lê é branca sobre preta, o formato do MNIST. Separar as duas evita que qualquer escolha visual acabe entrando no tensor.

**Média por área na redução, não `drawImage`.** Numa redução de ~14×, o reamostrador do navegador descarta pixels e o traço chega picotado à rede, o que muda bastante a previsão. O passo 3 faz a média dos pixels de cada célula de destino — o mesmo que o MNIST fez ao construir o dataset.

---