#!/usr/bin/env python3
"""Generate the fixed bilingual retrieval benchmark corpus and query set."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent

# Each topic has two anchor documents (one per language) and ten closely
# related distractors.  Queries are paraphrases and never contain document
# ids, titles, or verbatim anchor sentences.
TOPICS = [
    {
        "code": "01",
        "name": "residential battery storage",
        "en": "A lithium iron phosphate home battery is normally operated inside a conservative state-of-charge window. The battery management system balances cells and enforces a lower reserve so repeated deep discharge does not accelerate aging. Round-trip efficiency describes the energy lost between charging and later delivery.",
        "zh": "家用磷酸鐵鋰儲能電池通常在保守的荷電狀態範圍內運作。電池管理系統會平衡各單體並設定下限保留量，避免反覆深度放電加速老化。往返效率用來描述充電到再次供電之間的能量損失。",
        "queries": [
            ("zh", "en", "家用儲能電池如何減少深度循環造成的老化，並用什麼指標表示充放電損失？", "en_01_anchor"),
            ("en", "en", "How can a residential storage pack limit aging from deep cycling, and which metric describes charge-discharge losses?", "en_01_anchor"),
            ("zh", "zh", "說明住宅儲能電池的安全電量範圍、單體管理與能源損耗概念。", "zh_01_anchor"),
        ],
        "distractors": [
            ("en", "A photovoltaic inverter converts panel DC into grid-compatible AC and tracks the array's maximum power point; it does not by itself determine battery cell health."),
            ("zh", "太陽能逆變器把面板直流電轉成可併網的交流電，並追蹤最大功率點；它本身不負責判斷電池單體健康。"),
            ("en", "Lead-acid batteries tolerate neither frequent full discharge nor high ambient temperature well, so their maintenance schedule differs from that of modern lithium packs."),
            ("zh", "鉛酸電池通常不適合頻繁完全放電或高溫環境，因此維護方式和鋰電池不同。"),
            ("en", "A home energy controller can shift charging to sunny hours or cheap tariffs, while demand response changes when a load consumes electricity."),
            ("zh", "家庭能源控制器可以把充電移到日照充足或低電價時段；需求響應則是改變負載用電時間。"),
            ("en", "Grid frequency support asks an inverter to change its power quickly; the requested response is different from measuring stored energy."),
            ("zh", "電網頻率支援要求逆變器快速調整功率，這與量測電池剩餘能量是不同問題。"),
            ("en", "Battery thermal management uses sensors, airflow, or liquid plates to keep cells within a safe temperature range during high-power operation."),
            ("zh", "電池熱管理會使用感測器、氣流或液冷板，在高功率運轉時維持單體的安全溫度。"),
        ],
    },
    {
        "code": "02",
        "name": "Kubernetes rolling deployment",
        "en": "A Kubernetes Deployment performs a rolling update by creating a new ReplicaSet and gradually replacing pods. The maxUnavailable and maxSurge settings bound the rollout, while readiness probes keep a new pod out of service until it is ready. Revision history can be used to roll back.",
        "zh": "Kubernetes Deployment 會建立新的 ReplicaSet，逐步替換舊 Pod 來執行滾動更新。maxUnavailable 與 maxSurge 限制更新期間的數量，readiness probe 會在新 Pod 準備好前阻止流量送入；保留的修訂紀錄可用於回滾。",
        "queries": [
            ("zh", "en", "Kubernetes 服務如何逐步換掉舊 Pod，同時避免未就緒的新版接收流量，失敗時又如何退回？", "en_02_anchor"),
            ("en", "en", "How does a Kubernetes service replace pods gradually while protecting traffic from unready versions, and how can a failed release be undone?", "en_02_anchor"),
            ("zh", "zh", "解釋 Kubernetes Deployment 更新時的副本數量限制、就緒檢查和回復機制。", "zh_02_anchor"),
        ],
        "distractors": [
            ("en", "A Kubernetes Service supplies a stable virtual IP and selects pods by labels; it does not create a new ReplicaSet during a rollout."),
            ("zh", "Kubernetes Service 以標籤選取 Pod 並提供穩定虛擬 IP，但它不會在更新時建立新的 ReplicaSet。"),
            ("en", "A StatefulSet gives ordered identity and persistent volume association to stateful pods, unlike the interchangeable pods of a typical web Deployment."),
            ("zh", "StatefulSet 為有狀態 Pod 提供有序身分和持久磁碟關聯，和一般 Web Deployment 的可替換 Pod 不同。"),
            ("en", "Horizontal Pod Autoscaling changes replica count from observed metrics such as CPU or a custom request rate; it is separate from version replacement policy."),
            ("zh", "Horizontal Pod Autoscaling 依 CPU 或自訂請求率等指標改變副本數，和版本替換策略是分開的功能。"),
            ("en", "A liveness probe detects a stuck process and may restart it, whereas a readiness probe controls whether an otherwise running pod receives traffic."),
            ("zh", "liveness probe 用來發現卡住的程序並可能重啟；readiness probe 則控制仍在執行的 Pod 是否接收流量。"),
            ("en", "A container image digest identifies immutable image content, while a deployment revision records changes to the workload specification."),
            ("zh", "容器映像摘要識別不可變的映像內容，Deployment 修訂則記錄工作負載規格的改變。"),
        ],
    },
    {
        "code": "03",
        "name": "espresso extraction",
        "en": "Espresso extraction depends on grind size, dose, beverage ratio, water temperature, and contact time. A grind that is too fine raises resistance and can make the cup bitter, while a coarse grind may under-extract. The brew ratio and extraction yield are useful measurements when dialing in a recipe.",
        "zh": "義式濃縮的萃取取決於研磨粗細、粉量、飲品比例、水溫和接觸時間。研磨過細會提高阻力並可能使風味苦澀，過粗則可能萃取不足。調整配方時，沖煮比例和萃取率是有用的量測指標。",
        "queries": [
            ("zh", "en", "調整義式咖啡時，研磨阻力與粉水比例如何影響過萃或萃取不足，該量測什麼？", "en_03_anchor"),
            ("en", "en", "When dialing in espresso, how do grind resistance and the dose-to-drink ratio relate to over- or under-extraction, and what should be measured?", "en_03_anchor"),
            ("zh", "zh", "說明濃縮咖啡的研磨、粉量、水溫和接觸時間如何共同決定萃取結果。", "zh_03_anchor"),
        ],
        "distractors": [
            ("en", "Lightly roasted coffee beans are denser and often need a different grind or temperature than dark roasted beans, but roast color alone is not an extraction measurement."),
            ("zh", "淺焙咖啡豆密度較高，常需要不同研磨或水溫；不過烘焙顏色本身不是萃取率量測。"),
            ("en", "Pour-over brewing changes flow with a paper filter and a manually controlled pour, so its bed depth and contact pattern differ from espresso."),
            ("zh", "手沖咖啡使用濾紙並由人工控制注水，粉床深度和接觸方式都與濃縮不同。"),
            ("en", "Water alkalinity and hardness affect perceived acidity and scale formation in a machine; they are water chemistry variables rather than grinder settings."),
            ("zh", "水的鹼度與硬度會影響酸味感受和機器結垢，屬於水質變數而非磨豆機設定。"),
            ("en", "A bottomless portafilter helps reveal channeling patterns at the puck surface, but it does not directly report extraction yield."),
            ("zh", "無底把手能顯示粉餅表面的通道效應，但不會直接回報萃取率。"),
            ("en", "A coffee grinder's burr alignment and retention influence repeatability, while a scale is used to verify dose and beverage mass."),
            ("zh", "磨豆機刀盤對齊和殘粉會影響重現性，電子秤則用來核對粉量與成品重量。"),
        ],
    },
    {
        "code": "04",
        "name": "coral reef restoration",
        "en": "Coral reef restoration often attaches nursery-grown fragments to stable reef substrate. Teams monitor fragment survival, growth, and bleaching over time, while reducing local stress such as sediment and pollution. Mixing genotypes can improve resilience, but restoration cannot replace climate mitigation.",
        "zh": "珊瑚礁復育常把苗圃培養的珊瑚片段固定到穩定基質上。團隊會長期追蹤存活率、生長和白化，並降低沉積物與污染等局部壓力。混合不同基因型可能提高韌性，但復育不能取代氣候減緩。",
        "queries": [
            ("zh", "en", "珊瑚復育如何把培養片段固定到礁體並評估成效，還需要處理哪些局部壓力？", "en_04_anchor"),
            ("en", "en", "How are cultured coral fragments deployed and evaluated on a reef, and which local stressors must be reduced alongside the work?", "en_04_anchor"),
            ("zh", "zh", "描述珊瑚礁復育的移植、存活追蹤、基因多樣性和氣候限制。", "zh_04_anchor"),
        ],
        "distractors": [
            ("en", "Mangrove restoration reconnects tidal channels and establishes seedlings in intertidal mud; its success indicators include canopy development and sediment retention."),
            ("zh", "紅樹林復育會恢復潮汐水道並在潮間帶泥地種植幼苗，常用樹冠和沉積物留存評估。"),
            ("en", "A marine protected area can restrict fishing or anchoring over a large zone, but designation alone does not grow coral fragments."),
            ("zh", "海洋保護區可在大範圍限制捕魚或下錨，但劃設本身不會培養珊瑚片段。"),
            ("en", "Kelp forest projects transplant algae onto rocky coastline and watch herbivore pressure; the habitat-forming species is not a coral polyp."),
            ("zh", "海帶林計畫把藻類移植到岩岸並觀察植食壓力，形成棲地的物種不是珊瑚蟲。"),
            ("en", "Bleaching occurs when stressed corals lose symbiotic algae, so reducing heat exposure and monitoring temperature can complement field surveys."),
            ("zh", "珊瑚受壓而失去共生藻會出現白化，因此降低熱壓力並監測水溫可配合野外調查。"),
            ("en", "Seagrass restoration anchors shoots in shallow sediment and measures meadow coverage; its substrate and growth form differ from a reef colony."),
            ("zh", "海草復育把幼株固定在淺層沉積物並量測草床覆蓋率，其基質和生長型態不同於珊瑚群體。"),
        ],
    },
    {
        "code": "05",
        "name": "digital signatures",
        "en": "A digital signature is produced by signing a message hash with a private key and checked with the corresponding public key. It provides integrity and evidence of key possession, while a certificate links a public key to an identity. Encryption alone does not prove who sent a message, and compromised keys may need revocation.",
        "zh": "數位簽章是用私鑰簽署訊息雜湊，再用對應公鑰驗證。它可提供完整性與持有私鑰的證據，憑證則把公鑰連結到身分。單純加密無法證明訊息寄件者，私鑰外洩時可能需要撤銷。",
        "queries": [
            ("zh", "en", "公開金鑰系統怎麼驗證訊息沒有被改過並連結寄件者身分？若金鑰外洩該怎麼辦？", "en_05_anchor"),
            ("en", "en", "How does a public-key system verify that a message was not altered and associate it with a sender, and what happens after key compromise?", "en_05_anchor"),
            ("zh", "zh", "解釋數位簽章中雜湊、私鑰、公鑰、憑證和撤銷各自的作用。", "zh_05_anchor"),
        ],
        "distractors": [
            ("en", "TLS certificates authenticate a server during a connection and help negotiate encryption, but a certificate is not itself the signature on every application message."),
            ("zh", "TLS 憑證在連線時驗證伺服器並協助協商加密，但憑證本身不是每則應用訊息的簽章。"),
            ("en", "A password hash is a one-way verifier stored by a service; it is not a public key that anyone can use to verify a signed document."),
            ("zh", "密碼雜湊是服務端保存的單向驗證值，不是任何人都能用來驗證文件的公鑰。"),
            ("en", "HMAC authenticates a message with a shared secret, so both parties possess the same secret and a third party cannot distinguish the sender from the verifier."),
            ("zh", "HMAC 用共享祕密驗證訊息，雙方都持有同一祕密，第三方無法只靠它區分簽署者和驗證者。"),
            ("en", "Key rotation limits the lifetime of a credential and requires distributing a replacement key; rotation policy is different from hashing a payload."),
            ("zh", "金鑰輪替限制憑證的使用期間，也需要發佈替代金鑰；輪替政策和對內容做雜湊是不同事情。"),
            ("en", "A certificate authority publishes trust information and may expose revocation status through a list or an online protocol."),
            ("zh", "憑證授權中心發布信任資訊，也可能透過清單或線上協定提供撤銷狀態。"),
        ],
    },
    {
        "code": "06",
        "name": "bicycle derailleur indexing",
        "en": "Indexed bicycle shifting matches cable movement to the spacing between sprockets. Limit screws set the derailleur's endpoints, the barrel adjuster corrects small alignment errors, and a bent hanger can prevent accurate indexing even after cable tension is adjusted.",
        "zh": "有段變速自行車的變速定位會把拉線移動量對應到飛輪齒片間距。限位螺絲設定後變速器的端點，微調螺絲修正小幅對齊誤差；若吊耳彎曲，即使調整拉線張力也可能無法準確定位。",
        "queries": [
            ("zh", "en", "自行車有段變速無法準確對齒時，端點、微調和彎曲吊耳分別會造成什麼影響？", "en_06_anchor"),
            ("en", "en", "For inaccurate indexed bicycle shifting, what roles do the endpoints, barrel adjustment, and a bent hanger play?", "en_06_anchor"),
            ("zh", "zh", "說明變速定位如何配合齒片間距，以及限位螺絲、微調螺絲和吊耳的檢查順序。", "zh_06_anchor"),
        ],
        "distractors": [
            ("en", "Correct tire pressure changes rolling resistance and comfort; it does not set the cable movement used by an indexed derailleur."),
            ("zh", "正確胎壓會改變滾動阻力和舒適度，但不會設定有段變速器使用的拉線移動量。"),
            ("en", "A hydraulic disc brake transfers pressure through fluid to move a caliper, which is unrelated to the cable indexing of a gear changer."),
            ("zh", "油壓碟煞透過煞車油推動卡鉗，和變速器的拉線定位無關。"),
            ("en", "Cadence describes crank revolutions per minute and can guide gear choice, but it is not a mechanical adjustment of a derailleur."),
            ("zh", "迴轉速是曲柄每分鐘轉數，可用來選檔，卻不是變速器的機械調整項目。"),
            ("en", "Chain lubrication reduces friction and noise; excess lubricant can attract grit but cannot correct a misaligned hanger."),
            ("zh", "鏈條潤滑可降低摩擦和噪音，過多潤滑油會吸附砂粒，但不能修正歪掉的吊耳。"),
            ("en", "A cassette's tooth range affects climbing gears and derailleur capacity, while the exact click-to-sprocket relationship comes from indexing."),
            ("zh", "飛輪齒數範圍影響爬坡檔和變速器容量，而每一格撥桿對應齒片的關係來自定位系統。"),
        ],
    },
    {
        "code": "07",
        "name": "urban heat mitigation",
        "en": "Cool roofs reduce building heat gain by reflecting more sunlight and emitting infrared energy. Shade trees lower radiant exposure, while permeable surfaces can also manage stormwater. A city should evaluate albedo, shade distribution, nighttime temperatures, and maintenance rather than treating one surface as a universal fix.",
        "zh": "涼屋頂透過反射較多陽光並放出紅外線，降低建築吸熱。遮蔭樹木可減少輻射曝曬，透水鋪面還能管理雨水。城市評估時應同時考慮反照率、遮蔭分布、夜間溫度和維護，而非把單一表面當成萬用解方。",
        "queries": [
            ("zh", "en", "城市要降低建築和行人的熱暴露，屋頂反射、樹蔭與透水面各有什麼作用，如何評估？", "en_07_anchor"),
            ("en", "en", "How do reflective roofs, tree shade, and permeable ground reduce urban heat, and which outcomes should a city monitor?", "en_07_anchor"),
            ("zh", "zh", "整理涼屋頂的反照與散熱、樹蔭的輻射效果、透水鋪面的附帶效益和評估面向。", "zh_07_anchor"),
        ],
        "distractors": [
            ("en", "Green roofs add a vegetated layer that can cool through shading and evapotranspiration, but their irrigation and structural load differ from a reflective roof coating."),
            ("zh", "綠屋頂以植栽層透過遮蔭和蒸散降溫，但灌溉與結構荷重和反射塗層屋頂不同。"),
            ("en", "Air-conditioning removes indoor heat and may increase outdoor waste heat; it is an adaptation measure rather than a change to pavement reflectance."),
            ("zh", "冷氣把室內熱移出並可能增加戶外廢熱，屬於適應措施，不是改變路面反照率。"),
            ("en", "Urban ventilation studies examine wind corridors and building geometry, whereas albedo describes how much incoming sunlight a surface reflects."),
            ("zh", "都市通風研究看風廊和建築幾何，反照率則描述表面反射入射陽光的比例。"),
            ("en", "Drought-tolerant landscaping lowers irrigation demand; its shade benefit depends on plant maturity and canopy placement."),
            ("zh", "耐旱景觀可降低灌溉需求，遮蔭效益則取決於植株成熟度和樹冠位置。"),
            ("en", "A heat-warning system alerts residents using weather and health thresholds, but an alert does not modify a building's thermal properties."),
            ("zh", "高溫警報系統根據天氣與健康門檻通知居民，但警報不會改變建築熱性能。"),
        ],
    },
    {
        "code": "08",
        "name": "vaccine cold chain",
        "en": "Many refrigerated vaccines are stored at 2-8 degrees C, and freeze-sensitive products require protection from accidental freezing. Data loggers record temperature excursions; a shipment outside range should be quarantined and assessed using product-specific stability guidance before use or disposal.",
        "zh": "許多需冷藏的疫苗保存於攝氏 2 至 8 度，怕凍產品還必須避免意外結冰。資料記錄器會記下溫度偏離；運輸超出範圍時，應先隔離並依產品專屬穩定性指引評估，再決定使用或報廢。",
        "queries": [
            ("zh", "en", "疫苗冷鏈保存的溫度範圍和怕凍風險怎麼管理？發生溫度偏離後應採取什麼流程？", "en_08_anchor"),
            ("en", "en", "How should refrigerated vaccines be protected from freezing, and what process should follow a recorded temperature excursion?", "en_08_anchor"),
            ("zh", "zh", "解釋冷藏疫苗的溫度監測、結冰敏感性、隔離和穩定性判定。", "zh_08_anchor"),
        ],
        "distractors": [
            ("en", "Some frozen biological products are transported on dry ice, which is a different storage regime from a refrigerator monitored above the freezing point."),
            ("zh", "有些冷凍生物製品用乾冰運送，這是不同於冰點以上冰箱監測的保存制度。"),
            ("en", "A pharmacy inventory system tracks lot numbers and expiry dates, but those fields do not replace a temperature data logger."),
            ("zh", "藥局庫存系統記錄批號和有效期限，但這些欄位不能取代溫度資料記錄器。"),
            ("en", "Insulated packaging slows heat transfer during shipping; the validated pack-out and coolant placement determine how long the range is maintained."),
            ("zh", "保溫包裝減慢運輸中的熱傳，驗證過的裝箱方式和冷媒位置決定能維持範圍多久。"),
            ("en", "A vaccine vial monitor changes appearance with cumulative heat exposure, whereas an electronic logger records a time series of temperature."),
            ("zh", "疫苗瓶上的熱指示器反映累積熱暴露，電子記錄器則記下隨時間變化的溫度序列。"),
            ("en", "Immunization scheduling determines when doses are given and does not decide whether a vial that experienced a cold-chain excursion is usable."),
            ("zh", "接種排程決定何時施打劑量，不會決定經歷冷鏈偏離的瓶劑是否可用。"),
        ],
    },
    {
        "code": "09",
        "name": "river floodplain restoration",
        "en": "Floodplain restoration reconnects a river with seasonal wetlands so water can spread and slow during high flows. Setbacks or levee removal can lower peak levels, while hazard maps describe inundation and risk maps also account for people and assets. Native vegetation stabilizes soil and supports habitat.",
        "zh": "洪泛平原復育會讓河川重新連接季節性濕地，使高流量時的水可以展開並減速。退縮帶或拆除堤防可能降低洪峰；危害圖描述淹水範圍，風險圖還納入人口與資產。原生植被可穩定土壤並提供棲地。",
        "queries": [
            ("zh", "en", "恢復河川與濕地連結如何減緩洪峰？堤防退縮、地圖分類和植被各扮演什麼角色？", "en_09_anchor"),
            ("en", "en", "How does reconnecting a river to wetlands reduce flood peaks, and how do setbacks, maps, and vegetation contribute?", "en_09_anchor"),
            ("zh", "zh", "說明洪泛平原復育的蓄洪空間、堤防策略、危害與風險地圖及原生植被。", "zh_09_anchor"),
        ],
        "distractors": [
            ("en", "A reservoir dam stores water behind a wall and releases it through outlets; it changes a river hydrograph but does not recreate a floodplain wetland."),
            ("zh", "水庫大壩把水蓄在牆後並經由出水口釋放，會改變河川歷線但不會重建洪泛濕地。"),
            ("en", "Urban stormwater detention holds runoff in a basin and releases it later, usually on a developed catchment rather than a natural river corridor."),
            ("zh", "都市雨水滯洪池在開發集水區暫存逕流再延後排放，通常不是天然河廊。"),
            ("en", "Coastal surge barriers address saltwater flooding from tides and storms, a different driver from an inland river's seasonal high flow."),
            ("zh", "海岸風暴潮閘門處理潮汐和風暴帶來的海水淹水，成因不同於內陸河川季節性高流量。"),
            ("en", "A flood insurance rate reflects mapped exposure and policy rules; it does not by itself change channel capacity or wetland storage."),
            ("zh", "洪水保險費率反映圖上的暴露和保單規則，本身不會改變河道容量或濕地蓄水。"),
            ("en", "Stream gauges measure stage or discharge at a point, providing observations that can calibrate models but not replacing land-use planning."),
            ("zh", "水位站在特定位置量測水位或流量，可用來校準模型，但不能取代土地使用規劃。"),
        ],
    },
    {
        "code": "10",
        "name": "Zstandard compression",
        "en": "Zstandard combines LZ-style match finding with finite-state entropy coding. Higher compression levels usually spend more CPU to improve ratio, while decompression remains comparatively fast. A trained dictionary can substantially help small, repetitive records, and a checksum detects corruption rather than increasing compression.",
        "zh": "Zstandard 結合 LZ 類型的匹配搜尋與有限狀態熵編碼。較高壓縮等級通常花更多 CPU 來改善比例，而解壓仍相對快速。訓練過的字典能明顯幫助小型且重複的記錄；校驗和用來偵測損壞，不會提升壓縮率。",
        "queries": [
            ("zh", "en", "Zstandard 如何在尋找重複片段和熵編碼間取捨？高壓縮等級、字典與校驗和各影響什麼？", "en_10_anchor"),
            ("en", "en", "What trade-off does Zstandard make between match finding and entropy coding, and what do levels, dictionaries, and checksums affect?", "en_10_anchor"),
            ("zh", "zh", "整理 Zstandard 的匹配、熵編碼、壓縮等級、小資料字典和損壞偵測功能。", "zh_10_anchor"),
        ],
        "distractors": [
            ("en", "Gzip commonly uses DEFLATE, which combines LZ77 matching and Huffman coding; its command-line defaults and speed profile differ from Zstandard."),
            ("zh", "Gzip 常用 DEFLATE，結合 LZ77 匹配和 Huffman 編碼；其命令列預設與速度特性不同於 Zstandard。"),
            ("en", "Encryption hides content with a key but normally does not make data smaller; compression and confidentiality solve different problems."),
            ("zh", "加密用金鑰隱藏內容，通常不會縮小資料；壓縮和機密性解決的是不同問題。"),
            ("en", "Parquet stores columns with encodings and statistics so analytical engines can skip data; it is a file layout rather than one generic stream codec."),
            ("zh", "Parquet 以欄式布局、編碼和統計資訊服務分析引擎跳過資料，是檔案格式而非單一通用串流編解碼器。"),
            ("en", "A checksum or cryptographic hash detects accidental or intentional changes, but adding it does not improve the byte ratio of the compressed payload."),
            ("zh", "校驗和或密碼雜湊能偵測意外或惡意更改，但加入它不會改善壓縮資料的位元組比例。"),
            ("en", "Deduplication removes identical blocks before storage and can save space, while a compressor searches for patterns within the stream it receives."),
            ("zh", "去重會在儲存前移除相同區塊以節省空間，壓縮器則在收到的串流內搜尋模式。"),
        ],
    },
]

# Eight additional non-relevant notes per topic make 100 documents per
# language. They are distinct records, not copies used to pad the corpus.
EXPANSION_DISTRACTORS = {
    "01": [
        ("en", "A battery inverter may provide islanding protection and synchronize with the utility; those controls are separate from electrochemical capacity fade."),
        ("zh", "電池逆變器可能提供孤島保護並與公用電網同步，這些控制和電化學容量衰退是不同層次。"),
        ("en", "State of charge estimates combine current integration with voltage or model corrections, and an estimate can drift without calibration."),
        ("zh", "荷電狀態估計會結合電流積分與電壓或模型修正，未校準時估計值可能逐漸漂移。"),
        ("en", "A battery warranty may specify an end-of-life capacity threshold and cycle limits, whereas an operating reserve is a dispatch setting."),
        ("zh", "電池保固可能規定壽命終點容量門檻和循環限制，而運轉保留量是調度設定。"),
        ("en", "Peak shaving reduces a site's highest grid draw by scheduling flexible loads; it does not define how cells are balanced."),
        ("zh", "削峰透過安排彈性負載降低場所的最高電網取電量，但不會定義單體如何平衡。"),
    ],
    "02": [
        ("en", "A PodDisruptionBudget limits voluntary disruptions during maintenance, but it does not set how a new ReplicaSet replaces an old one."),
        ("zh", "PodDisruptionBudget 限制維護時的自願中斷，但不會設定新 ReplicaSet 如何取代舊版本。"),
        ("en", "A namespace scopes names and policies for objects, while an individual deployment still owns its own pod template and revisions."),
        ("zh", "Namespace 為物件劃分名稱和政策範圍，但單一 Deployment 仍擁有自己的 Pod 範本與修訂。"),
        ("en", "A ConfigMap supplies non-secret configuration to a container; changing it may trigger an application reload without being a rollout strategy."),
        ("zh", "ConfigMap 將非祕密設定提供給容器；變更它可能觸發應用程式重新載入，但不是滾動更新策略。"),
        ("en", "A node taint and pod toleration influence placement decisions, not the readiness condition that controls service traffic."),
        ("zh", "節點 taint 和 Pod toleration 影響排程位置，不是控制服務流量的 readiness 條件。"),
    ],
    "03": [
        ("en", "A dosing funnel helps distribute grounds into a portafilter, and a tamper makes the bed level; neither one measures dissolved solids."),
        ("zh", "粉碗漏斗幫助把咖啡粉分布進濾杯，填壓器讓粉床平整；兩者都不會量測溶解固形物。"),
        ("en", "A thermoblock or boiler determines how water is heated, while a flow-control valve changes the pressure profile during a shot."),
        ("zh", "加熱塊或鍋爐決定水如何加熱，流量控制閥則改變一杯咖啡過程中的壓力曲線。"),
        ("en", "Freshly roasted beans release carbon dioxide during a rest period, which can affect crema and puck behavior before recipe tuning."),
        ("zh", "新鮮烘焙豆在養豆期間釋放二氧化碳，可能影響油脂與粉餅行為，之後才適合調配方。"),
        ("en", "A refractometer estimates beverage total dissolved solids and can be combined with beverage mass to calculate extraction, rather than replacing grind adjustment."),
        ("zh", "折射計估計飲品總溶解固形物，可和成品重量合併計算萃取率，但不能取代研磨調整。"),
    ],
    "04": [
        ("en", "A reef survey may photograph transects and classify benthic cover; a photo record is evidence for monitoring, not a transplant method."),
        ("zh", "礁體調查可拍攝樣線並分類底棲覆蓋；影像紀錄是監測證據，不是移植方法。"),
        ("en", "Water-quality sampling can track nutrients and turbidity near a reef, while a genetic assay examines lineage or symbiont composition."),
        ("zh", "水質採樣可追蹤礁區附近的營養鹽和濁度，基因分析則檢查譜系或共生體組成。"),
        ("en", "An artificial reef changes available hard structure for fish, but its material and placement do not automatically establish living coral."),
        ("zh", "人工魚礁增加魚類可用的硬質結構，但材料和位置不會自動形成活珊瑚。"),
        ("en", "A bleaching alert can trigger a temporary closure or survey; the alert threshold is an operational decision rather than a restoration technique."),
        ("zh", "白化警報可觸發臨時封閉或調查；警報門檻是運作決策，不是復育技術。"),
    ],
    "05": [
        ("en", "A key exchange protocol establishes a shared secret over an untrusted network, but it does not by itself sign an application payload."),
        ("zh", "金鑰交換協定可在不可信網路建立共享祕密，但本身不會替應用資料簽章。"),
        ("en", "A nonce prevents replay in a protocol message; uniqueness is its purpose, unlike a digest's role in detecting content changes."),
        ("zh", "Nonce 讓協定訊息避免重放，重點是唯一性，不同於摘要用來發現內容改變。"),
        ("en", "A hardware security module protects private-key operations and can enforce policy, while public verification can still happen on an ordinary host."),
        ("zh", "硬體安全模組保護私鑰操作並可執行政策，而公鑰驗證仍可在一般主機上進行。"),
        ("en", "A signed software package often includes metadata about version and dependencies; package metadata does not replace verification of the signature."),
        ("zh", "已簽署的軟體套件常附帶版本和相依性中繼資料，但套件中繼資料不能取代簽章驗證。"),
    ],
    "06": [
        ("en", "A chain checker measures elongation from wear and can indicate replacement time, but it does not set the derailleur's cable ratio."),
        ("zh", "鏈條尺可從磨損量測伸長並提示更換時機，但不會設定變速器的拉線比例。"),
        ("en", "Crank length changes leverage and fit, while cassette spacing determines the geometry that an indexed shifter must match."),
        ("zh", "曲柄長度改變槓桿和騎乘適配，飛輪間距則決定有段變速必須配合的幾何。"),
        ("en", "Cable housing with contamination can add friction and delay a shift; cleaning or replacement addresses the housing rather than a bent frame part."),
        ("zh", "受污染的變速外管會增加摩擦並延遲換檔，清潔或更換是處理外管，不是修理彎曲車架零件。"),
        ("en", "A front derailleur moves the chain between chainrings and has its own height and limit setup, distinct from the rear cassette indexing described here."),
        ("zh", "前變速器把鏈條移到不同大盤，具有自己的高度與限位設定，和後飛輪定位不同。"),
    ],
    "07": [
        ("en", "Urban tree planting needs adequate soil volume and irrigation during establishment; survival planning is not the same as measuring roof reflectance."),
        ("zh", "都市植樹需要足夠土壤體積並在建立期灌溉，存活規劃不等於量測屋頂反射。"),
        ("en", "Pavement temperature varies with material, shade, moisture, and traffic; one midday reading cannot represent neighborhood heat exposure."),
        ("zh", "路面溫度受材料、遮蔭、含水量和車流影響，單次中午讀值不能代表整個街區的熱暴露。"),
        ("en", "Building insulation slows heat transfer through walls and ceilings, while a cool roof primarily changes the energy absorbed at the outer surface."),
        ("zh", "建築隔熱減慢牆和天花板的熱傳，涼屋頂主要改變外表面吸收的能量。"),
        ("en", "Public cooling centers provide refuge during a heat event; their operation does not alter albedo or stormwater infiltration."),
        ("zh", "公共避暑中心在高溫事件提供休息場所，但營運不會改變反照率或雨水入滲。"),
    ],
    "08": [
        ("en", "A refrigerator alarm can notify staff when a threshold is crossed, but response still requires checking the recorded duration and product guidance."),
        ("zh", "冰箱警報可在超過門檻時通知人員，但處置仍需查核偏離持續時間和產品指引。"),
        ("en", "A shipment label identifies handling instructions and destination, whereas a temperature logger provides evidence of the actual excursion."),
        ("zh", "運輸標籤標示處理指示和目的地，溫度記錄器則提供實際偏離的證據。"),
        ("en", "Stock rotation uses expiry dates or receipt order to reduce waste, independent of whether a cold-chain monitor was activated."),
        ("zh", "庫存輪替依有效日期或進貨順序減少浪費，與冷鏈監測器是否啟動無關。"),
        ("en", "A clinic refrigerator should avoid crowding vents and keep an inventory map; arrangement helps airflow but cannot validate an excursion retrospectively."),
        ("zh", "診所冰箱應避免堵住出風口並保留庫存位置圖，排列有助氣流但不能事後驗證溫度偏離。"),
    ],
    "09": [
        ("en", "A channel restoration project may add large wood to diversify habitat, but a single log does not provide the storage area of a connected floodplain."),
        ("zh", "河道復育可加入大型木材增加棲地多樣性，但單根木頭沒有連通洪泛平原的蓄水空間。"),
        ("en", "A rainfall forecast gives advance warning for a catchment, while a floodplain map represents where water is expected to spread."),
        ("zh", "降雨預報為集水區提供提前警示，洪泛平原地圖則表示水預期會在哪裡展開。"),
        ("en", "Bank armoring can protect a particular property but may transfer erosion downstream, unlike a setback that leaves room for lateral movement."),
        ("zh", "護岸可保護特定財產但可能把侵蝕轉移到下游，不同於保留側向移動空間的退縮帶。"),
        ("en", "A drought plan manages low-flow water allocation and ecological releases; it addresses scarcity rather than the storage response to a flood peak."),
        ("zh", "乾旱計畫管理低流量時的用水分配和生態放流，處理的是缺水而非洪峰蓄水反應。"),
    ],
    "10": [
        ("en", "A compressor frame can split a stream into independently checksummed blocks, allowing partial recovery without changing the core entropy coder."),
        ("zh", "壓縮框架可把串流切成各自有校驗和的區塊，便於部分復原而不改變核心熵編碼器。"),
        ("en", "A benchmark of compression should report input type, level, CPU time, and output size because a ratio on one dataset is not universal."),
        ("zh", "壓縮基準測試應回報輸入類型、等級、CPU 時間和輸出大小，因為單一資料集的比例不是通用結果。"),
        ("en", "Streaming decompression can process a frame incrementally, while random access requires framing or an index around the compressed stream."),
        ("zh", "串流解壓可逐步處理框架，而隨機存取需要在壓縮串流外加框架或索引。"),
        ("en", "A dictionary must be available to the decoder and should be trained on representative samples; an arbitrary dictionary can make output worse."),
        ("zh", "解碼器必須取得字典，且字典應以代表性樣本訓練；任意字典可能讓輸出更差。"),
    ],
}


def write_jsonl(path: Path, rows: list[dict]) -> str:
    text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    documents: list[dict] = []
    queries: list[dict] = []
    for topic in TOPICS:
        code = topic["code"]
        documents.append({"id": f"en_{code}_anchor", "language": "en", "topic": topic["name"], "text": topic["en"]})
        documents.append({"id": f"zh_{code}_anchor", "language": "zh", "topic": topic["name"], "text": topic["zh"]})
        distractors = topic["distractors"] + EXPANSION_DISTRACTORS[code]
        for idx, (lang, text) in enumerate(distractors, 1):
            documents.append({"id": f"{lang}_{code}_d{idx:02d}", "language": lang, "topic": topic["name"], "text": text})
        for qidx, (qlang, dlang, query, target) in enumerate(topic["queries"], 1):
            queries.append({
                "id": f"q{code}_{qidx}",
                "query": query,
                "language": qlang,
                "target_language": dlang,
                "target_doc_id": target,
                "topic": topic["name"],
                "direction": f"{qlang}->{dlang}",
            })
    assert len(documents) == 200, len(documents)
    assert len(queries) == 30, len(queries)
    corpus_hash = write_jsonl(ROOT / "corpus.jsonl", documents)
    query_hash = write_jsonl(ROOT / "queries.jsonl", queries)
    manifest = {
        "corpus_file": "corpus.jsonl",
        "queries_file": "queries.jsonl",
        "documents": len(documents),
        "documents_by_language": {"en": sum(doc["language"] == "en" for doc in documents), "zh": sum(doc["language"] == "zh" for doc in documents)},
        "queries": len(queries),
        "topics": len(TOPICS),
        "sha256": {"corpus.jsonl": corpus_hash, "queries.jsonl": query_hash},
        "query_directions": {"zh->en": 10, "en->en": 10, "zh->zh": 10},
        "relevance": "Exactly one anchor document is relevant to each query; target_doc_id is the judged relevant id.",
    }
    (ROOT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
