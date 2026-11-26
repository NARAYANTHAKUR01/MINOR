"""
NLP Object Extractor for Voice and Text Queries.

Extracts COCO 80 object classes from natural speech-to-text sentences,
supporting:
- All 80 COCO classes with exact ID mapping (0 to 79)
- Longest-phrase matching first (e.g. 'hot dog' before 'dog', 'teddy bear' before 'bear')
- Comprehensive colloquial aliases, slang, regional dialects, and spoken synonyms
- Pre-processing phonetic and sound-alike speech normalizations
- Morphological lemmatization & irregular plural handling
- STT error tolerance via tuned fuzzy matching (difflib)
- Context-aware negation detection ('find the person, not the car')
- Resetting negation boundary on contrasting conjunctions ('not the cat, but the dog')
- Intent classification ('locate', 'count', 'detect', 'filter', 'general')
- Quantity & modifier/color/material extraction ('two red apples', 'a wooden dining table')
"""

import difflib
import re
import time
from typing import Any, Literal

from backend.schemas.nlp import (
    ClassDetail,
    ExtractedObject,
    NLPExtractionResponse,
    SupportedClassesResponse,
)

# ---------------------------------------------------------------------------
# 80 Canonical COCO Classes in exact ID order (0 to 79)
# ---------------------------------------------------------------------------
COCO_CLASSES: list[str] = [
    "person",         # 0
    "bicycle",        # 1
    "car",            # 2
    "motorcycle",     # 3
    "airplane",       # 4
    "bus",            # 5
    "train",          # 6
    "truck",          # 7
    "boat",           # 8
    "traffic light",  # 9
    "fire hydrant",   # 10
    "stop sign",      # 11
    "parking meter",  # 12
    "bench",          # 13
    "bird",           # 14
    "cat",            # 15
    "dog",            # 16
    "horse",          # 17
    "sheep",          # 18
    "cow",            # 19
    "elephant",       # 20
    "bear",           # 21
    "zebra",          # 22
    "giraffe",        # 23
    "backpack",       # 24
    "umbrella",       # 25
    "handbag",        # 26
    "tie",            # 27
    "suitcase",       # 28
    "frisbee",        # 29
    "skis",           # 30
    "snowboard",      # 31
    "sports ball",    # 32
    "kite",           # 33
    "baseball bat",   # 34
    "baseball glove", # 35
    "skateboard",     # 36
    "surfboard",      # 37
    "tennis racket",  # 38
    "bottle",         # 39
    "wine glass",     # 40
    "cup",            # 41
    "fork",           # 42
    "knife",          # 43
    "spoon",          # 44
    "bowl",           # 45
    "banana",         # 46
    "apple",          # 47
    "sandwich",       # 48
    "orange",         # 49
    "broccoli",       # 50
    "carrot",         # 51
    "hot dog",        # 52
    "pizza",          # 53
    "donut",          # 54
    "cake",           # 55
    "chair",          # 56
    "couch",          # 57
    "potted plant",   # 58
    "bed",            # 59
    "dining table",   # 60
    "toilet",         # 61
    "tv",             # 62
    "laptop",         # 63
    "mouse",          # 64
    "remote",         # 65
    "keyboard",       # 66
    "cell phone",     # 67
    "microwave",      # 68
    "oven",           # 69
    "toaster",        # 70
    "sink",           # 71
    "refrigerator",   # 72
    "book",           # 73
    "clock",          # 74
    "vase",           # 75
    "scissors",       # 76
    "teddy bear",     # 77
    "hair drier",     # 78
    "toothbrush",     # 79
]

CLASS_NAME_TO_ID: dict[str, int] = {name: idx for idx, name in enumerate(COCO_CLASSES)}

# ---------------------------------------------------------------------------
# Phonetic Sound-Alike & Speech Transcription Normalization
# ---------------------------------------------------------------------------
PHONETIC_NORMALIZATION: dict[str, str] = {
    "sell phone": "cell phone",
    "sellphone": "cell phone",
    "cellfone": "cell phone",
    "cellfones": "cell phones",
    "smartfone": "smartphone",
    "smartfones": "smartphones",
    "aeroplan": "airplane",
    "aeroplain": "airplane",
    "telivision": "television",
    "televison": "television",
    "refregerator": "refrigerator",
    "refridgerator": "refrigerator",
    "refrigirator": "refrigerator",
    "brocolli": "broccoli",
    "brocoli": "broccoli",
    "brockoli": "broccoli",
    "sandwhich": "sandwich",
    "sandwitch": "sandwich",
    "pisa": "pizza",
    "piza": "pizza",
    "hambuger": "hamburger",
    "hotdog": "hot dog",
    "hotdogs": "hot dogs",
    "teddybear": "teddy bear",
    "teddybears": "teddy bears",
    "stopsign": "stop sign",
    "trafficlight": "traffic light",
    "firehydrant": "fire hydrant",
    "parkingmeter": "parking meter",
    "wineglass": "wine glass",
    "wineglasses": "wine glasses",
    "hairdryer": "hair drier",
    "hair-dryer": "hair drier",
    "hair dryer": "hair drier",
    "hair dryers": "hair driers",
    "blowdryer": "hair drier",
    "blow-dryer": "hair drier",
    "blow dryer": "hair drier",
    "diningtable": "dining table",
    "dinnertable": "dining table",
    "waterbottle": "water bottle",
    "waterbottles": "water bottles",
    "sportsball": "sports ball",
    "sportsballs": "sports balls",
    "baseballbat": "baseball bat",
    "baseballglove": "baseball glove",
    "tennisracket": "tennis racket",
    "tennisracquet": "tennis racket",
    "tooth brush": "toothbrush",
    "tooth brushes": "toothbrushes",
    "pottedplant": "potted plant",
    "pottedplants": "potted plants",
    "remotecontrol": "remote control",
    "tvremote": "tv remote",
    "back pack": "backpack",
    "back packs": "backpacks",
    "hand bag": "handbag",
    "hand bags": "handbags",
    "suit case": "suitcase",
    "suit cases": "suitcases",
    "snow board": "snowboard",
    "skate board": "skateboard",
    "surf board": "surfboard",
    "microwave oven": "microwave",
    "micro wave": "microwave",
    "mac book": "macbook",
    "macbooks": "macbooks",
    "i phone": "iphone",
    "iphones": "iphones",
    "two wheeler": "motorcycle",
    "two-wheeler": "motorcycle",
    "living room couch": "couch",
    # Additional high-frequency speech & STT sound-alike normalizations
    "lap atop": "laptop",
    "lab top": "laptop",
    "labtop": "laptop",
    "study bear": "teddy bear",
    "tee vee": "tv",
    "teevee": "tv",
    "tee-vee": "tv",
    "t v": "tv",
    "t.v.": "tv",
    "botle": "bottle",
    "bottl": "bottle",
    "water botle": "water bottle",
    "waterbottel": "water bottle",
    "sissors": "scissors",
    "scisors": "scissors",
    "cissors": "scissors",
    "assiserus": "scissors",
    "bycycle": "bicycle",
    "bicicle": "bicycle",
    "motocycle": "motorcycle",
    "aeroplane": "airplane",
    "air plane": "airplane",
    "chare": "chair",
    "cheir": "chair",
    "sofa bed": "couch",
    "sofaset": "couch",
    "coffe mug": "cup",
    "dinning table": "dining table",
    "key board": "keyboard",
    "key-board": "keyboard",
    "compuer": "computer",
    "moniter": "monitor",
    "screne": "screen",
    "bannana": "banana",
    "bananna": "banana",
    "orrange": "orange",
    "appel": "apple",
    "aple": "apple",
    "bagpack": "backpack",
    "bag pack": "backpack",
    "book bag": "backpack",
    "bookbag": "backpack",
    "day pack": "backpack",
    "sholder bag": "handbag",
    "carry bag": "backpack",
    "shopping bag": "handbag",
}

# ---------------------------------------------------------------------------
# Comprehensive Aliases & Spoken Synonyms Map across ALL 80 COCO Classes
# Maps colloquial / spoken terms -> Canonical COCO class name
# ---------------------------------------------------------------------------
CLASS_ALIASES: dict[str, list[str]] = {
    "person": [
        "person", "people", "human", "humans", "man", "men", "woman", "women",
        "child", "children", "kid", "kids", "guy", "guys", "boy", "boys",
        "girl", "girls", "baby", "babies", "toddler", "toddlers", "pedestrian",
        "pedestrians", "someone", "somebody", "anybody", "anyone", "lady",
        "ladies", "gentleman", "gentlemen", "individual", "individuals", "cop",
        "cops", "police", "policeman", "policewoman", "cyclist", "cyclists",
        "driver", "drivers", "passenger", "passengers", "worker", "workers",
        "spectator", "spectators", "bystander", "bystanders", "stranger",
        "folks", "folk", "buddy", "friend", "dude", "dudes", "inhabitant"
    ],
    "bicycle": [
        "bicycle", "bicycles", "bike", "bikes", "cycle", "cycles",
        "pushbike", "pushbikes", "bicycling", "cycling", "road bike",
        "mountain bike", "mountainbike", "bmx", "tandem", "e-bike", "ebike",
        "electric bike", "velocipede", "two-wheeler", "cycle rickshaw"
    ],
    "car": [
        "car", "cars", "automobile", "automobiles", "vehicle", "vehicles",
        "sedan", "sedans", "suv", "suvs", "cab", "cabs", "taxi", "taxis",
        "coupe", "coupes", "motorcar", "motorcars", "auto", "autos",
        "hatchback", "hatchbacks", "convertible", "convertibles", "wagon",
        "station wagon", "minivan", "minivans", "jeep", "jeeps", "ev",
        "electric car", "sports car", "roadster", "motor vehicle", "passenger car"
    ],
    "motorcycle": [
        "motorcycle", "motorcycles", "motorbike", "motorbikes", "scooter",
        "scooters", "moped", "mopeds", "chopper", "choppers", "dirtbike",
        "dirt bike", "dirtbikes", "vespa", "vespas", "motor bike",
        "motor bikes", "superbike", "superbikes", "harley", "harleys"
    ],
    "airplane": [
        "airplane", "airplanes", "aeroplane", "aeroplanes", "plane", "planes",
        "jet", "jets", "aircraft", "airliner", "airliners", "jumbo jet",
        "flight", "flights", "passenger plane", "cargo plane", "jetliner",
        "biplane", "seaplane", "glider", "airbus", "boeing"
    ],
    "bus": [
        "bus", "buses", "busses", "autobus", "omnibus", "schoolbus",
        "school bus", "school buses", "transit bus", "coach", "coaches",
        "shuttle", "shuttles", "double decker", "double-decker",
        "trolleybus", "city bus", "motorcoach"
    ],
    "train": [
        "train", "trains", "locomotive", "locomotives", "railcar", "railcars",
        "metro", "subway", "subway train", "freight train", "bullet train",
        "tram", "trams", "streetcar", "passenger train", "train car", "rail",
        "railway", "monorail", "cargo train", "express train", "tube train"
    ],
    "truck": [
        "truck", "trucks", "pickup", "pickup truck", "pickup trucks", "pickups",
        "lorry", "lorries", "semi", "semi-truck", "semi truck", "semi trucks",
        "trailer", "18-wheeler", "eighteen wheeler", "tanker", "haul truck",
        "dump truck", "delivery truck", "box truck", "flatbed", "tow truck",
        "fire truck", "garbage truck", "freight truck", "dumper", "hauler"
    ],
    "boat": [
        "boat", "boats", "ship", "ships", "vessel", "vessels", "yacht",
        "yachts", "speedboat", "speedboats", "sailboat", "sailboats", "canoe",
        "canoes", "kayak", "kayaks", "raft", "rafts", "ferry", "ferries",
        "rowboat", "rowboats", "dinghy", "dinghies", "cruiser", "trawler",
        "catamaran", "pontoon", "barge", "tugboat", "watercraft", "motorboat",
        "steamship", "schooner", "gondola"
    ],
    "traffic light": [
        "traffic light", "traffic lights", "traffic signal", "traffic signals",
        "stoplight", "stoplights", "stop light", "stop lights", "red light",
        "green light", "yellow light", "signal light", "semaphore"
    ],
    "fire hydrant": [
        "fire hydrant", "fire hydrants", "hydrant", "hydrants", "fireplug",
        "fireplugs", "fire plug", "water hydrant", "water hydrants"
    ],
    "stop sign": [
        "stop sign", "stop signs", "octagonal sign", "red stop sign",
        "stop-sign", "stop-signs"
    ],
    "parking meter": [
        "parking meter", "parking meters", "parking ticket machine",
        "parking machine", "ticket meter"
    ],
    "bench": [
        "bench", "benches", "park bench", "park benches", "wooden bench",
        "pew", "pews", "bleacher", "bleachers", "park seat"
    ],
    "bird": [
        "bird", "birds", "pigeon", "pigeons", "sparrow", "sparrows", "crow",
        "crows", "seagull", "seagulls", "duck", "ducks", "hawk", "hawks",
        "eagle", "eagles", "parrot", "parrots", "owl", "owls", "goose",
        "geese", "swan", "swans", "raven", "ravens", "robin", "robins",
        "bluejay", "finch", "finches", "dove", "doves", "canary", "canaries",
        "swallow", "heron", "flamingo", "woodpecker", "falcon", "birdy",
        "birdie", "fowl", "avian", "parakeet", "cockatoo", "pelican"
    ],
    "cat": [
        "cat", "cats", "kitten", "kittens", "kitty", "kitties", "feline",
        "felines", "tomcat", "tomcats", "pussycat", "pussycats", "tabby",
        "tabbies", "calico", "siamese"
    ],
    "dog": [
        "dog", "dogs", "puppy", "puppies", "pup", "pups", "hound", "hounds",
        "canine", "canines", "pooch", "pooches", "doggie", "doggies", "doggy",
        "retriever", "golden retriever", "labrador", "german shepherd",
        "bulldog", "poodle", "beagle", "husky", "rottweiler", "terrier",
        "mutt", "mutts", "dachshund", "pug", "pugs", "boxer", "chihuahua"
    ],
    "horse": [
        "horse", "horses", "stallion", "stallions", "mare", "mares", "pony",
        "ponies", "colt", "colts", "foal", "foals", "mustang", "mustangs",
        "steed", "steeds", "filly", "bronco", "equine", "racehorse"
    ],
    "sheep": [
        "sheep", "lamb", "lambs", "ewe", "ewes", "ram", "rams", "lambkin", "fleece"
    ],
    "cow": [
        "cow", "cows", "bull", "bulls", "calf", "calves", "cattle", "bovine",
        "ox", "oxen", "heifer", "heifers", "dairy cow", "beef cattle", "steer"
    ],
    "elephant": [
        "elephant", "elephants", "tusker", "tuskers", "pachyderm", "baby elephant"
    ],
    "bear": [
        "bear", "bears", "grizzly", "grizzlies", "polar bear", "black bear",
        "brown bear", "panda", "pandas", "bear cub"
    ],
    "zebra": [
        "zebra", "zebras"
    ],
    "giraffe": [
        "giraffe", "giraffes"
    ],
    "backpack": [
        "backpack", "backpacks", "bag", "bags", "bagpack", "bagpacks", "bag pack", "bag packs",
        "rucksack", "rucksacks", "knapsack", "knapsacks", "schoolbag", "schoolbags",
        "school bag", "school bags", "bookbag", "book bag", "daypack", "haversack",
        "pack", "packs", "carry bag", "shopping bag"
    ],
    "umbrella": [
        "umbrella", "umbrellas", "brolly", "brollies", "parasol", "parasols",
        "rain umbrella", "sun umbrella"
    ],
    "handbag": [
        "handbag", "handbags", "purse", "purses", "pocketbook", "pocketbooks",
        "clutch", "clutches", "tote bag", "tote bags", "tote", "totes",
        "shoulder bag", "satchel", "satchels", "wristlet", "pocket book",
        "wallet", "wallets", "small bag"
    ],
    "tie": [
        "tie", "ties", "necktie", "neckties", "bow tie", "bow ties",
        "bowtie", "bowties", "bow-tie", "cravat", "neck tie"
    ],
    "suitcase": [
        "suitcase", "suitcases", "luggage", "baggages", "baggage", "valise",
        "valises", "travel bag", "carry-on", "carry on", "carryon", "trunk",
        "trunks", "trolley bag", "rolling bag", "duffel", "duffle", "duffel bag"
    ],
    "frisbee": [
        "frisbee", "frisbees", "flying disc", "flying discs", "ultimate disc",
        "frisby", "frizbee"
    ],
    "skis": [
        "skis", "ski", "pair of skis", "snow skis", "skiing", "ski boards"
    ],
    "snowboard": [
        "snowboard", "snowboards", "snow board", "snow boards", "snowboarding"
    ],
    "sports ball": [
        "sports ball", "sports balls", "ball", "balls", "soccer ball",
        "soccer balls", "football", "footballs", "basketball", "basketballs",
        "tennis ball", "tennis balls", "volleyball", "volleyballs", "rugby ball",
        "golf ball", "baseball", "baseballs", "softball", "bowling ball"
    ],
    "kite": [
        "kite", "kites", "box kite", "stunt kite"
    ],
    "baseball bat": [
        "baseball bat", "baseball bats", "wooden bat", "metal bat",
        "softball bat", "slugger"
    ],
    "baseball glove": [
        "baseball glove", "baseball gloves", "baseball mitt", "baseball mitts",
        "catcher's mitt", "ball glove", "fielding glove"
    ],
    "skateboard": [
        "skateboard", "skateboards", "skate board", "skate boards",
        "longboard", "longboards", "penny board", "skateboarding"
    ],
    "surfboard": [
        "surfboard", "surfboards", "surf board", "surf boards",
        "wave board", "bodyboard", "surfing board"
    ],
    "tennis racket": [
        "tennis racket", "tennis rackets", "tennis racquet", "tennis racquets",
        "racquet", "racquets", "racket", "rackets", "squash racket", "badminton racket"
    ],
    "bottle": [
        "bottle", "bottles", "water bottle", "water bottles", "waterbottle", "waterbottles",
        "plastic bottle", "glass bottle", "flask", "flasks", "thermos", "thermoses",
        "jug", "jugs", "canteen", "canteens", "beverage bottle", "soda bottle",
        "wine bottle", "beer bottle", "can", "cans", "soda can", "tin", "tins", "sipper", "water flask"
    ],
    "wine glass": [
        "wine glass", "wine glasses", "glass", "glasses", "drinking glass", "drinking glasses",
        "water glass", "water glasses", "goblet", "goblets", "champagne flute",
        "champagne flutes", "wineglass", "wineglasses", "chalice", "stemware", "wine cup",
        "tumbler glass", "whiskey glass", "beer glass", "pint glass"
    ],
    "cup": [
        "cup", "cups", "mug", "mugs", "coffee mug", "coffee mugs", "coffee cup",
        "coffee cups", "teacup", "teacups", "tea cup", "tumbler", "tumblers",
        "drinking cup", "disposable cup", "paper cup", "espresso cup"
    ],
    "fork": [
        "fork", "forks", "dinner fork", "salad fork", "dessert fork", "plastic fork", "cutlery", "eating utensil"
    ],
    "knife": [
        "knife", "knives", "pocketknife", "pocketknives", "blade", "blades",
        "butter knife", "steak knife", "kitchen knife", "chef knife", "chef's knife",
        "carving knife", "pocket knife", "cutlery knife", "flatware", "silverware"
    ],
    "spoon": [
        "spoon", "spoons", "teaspoon", "teaspoons", "tablespoon", "tablespoons",
        "soup spoon", "soup spoons", "dessert spoon"
    ],
    "bowl": [
        "bowl", "bowls", "soup bowl", "salad bowl", "cereal bowl",
        "mixing bowl", "dish bowl", "fruit bowl", "plate", "plates", "dinner plate",
        "dinner plates", "dish", "dishes", "saucer", "saucers", "platter", "platters", "tray", "trays"
    ],
    "banana": [
        "banana", "bananas", "plantain", "plantains"
    ],
    "apple": [
        "apple", "apples", "red apple", "green apple", "granny smith", "gala apple"
    ],
    "sandwich": [
        "sandwich", "sandwiches", "burger", "burgers", "hamburger",
        "hamburgers", "cheeseburger", "cheeseburgers", "sub", "subs",
        "hoagie", "hoagies", "panini", "paninis", "panino", "wrap", "wraps",
        "bagel sandwich", "grilled cheese", "club sandwich"
    ],
    "orange": [
        "orange", "oranges", "clementine", "clementines", "tangerine",
        "tangerines", "mandarin", "mandarins", "citrus"
    ],
    "broccoli": [
        "broccoli", "broccolis", "florets", "broccoli florets"
    ],
    "carrot": [
        "carrot", "carrots", "baby carrot", "baby carrots"
    ],
    "hot dog": [
        "hot dog", "hot dogs", "frankfurter", "frankfurters", "wiener",
        "wieners", "corn dog", "sausage in a bun"
    ],
    "pizza": [
        "pizza", "pizzas", "pizza slice", "slice of pizza", "pepperoni pizza",
        "cheese pizza", "pizza pie"
    ],
    "donut": [
        "donut", "donuts", "doughnut", "doughnuts", "glazed donut", "frosted donut"
    ],
    "cake": [
        "cake", "cakes", "pastry", "pastries", "cupcake", "cupcakes",
        "cheesecake", "cheesecakes", "birthday cake", "chocolate cake",
        "wedding cake", "tart", "tarts", "muffin", "muffins", "slice of cake"
    ],
    "chair": [
        "chair", "chairs", "armchair", "armchairs", "seat", "seats",
        "stool", "stools", "desk chair", "office chair", "rocking chair",
        "folding chair", "swivel chair", "dining chair"
    ],
    "couch": [
        "couch", "couches", "sofa", "sofas", "settee", "settees",
        "davenport", "loveseat", "loveseats", "lounge", "futon",
        "sectional", "daybed", "chesterfield", "sofa bed", "couch sofa"
    ],
    "potted plant": [
        "potted plant", "potted plants", "houseplant", "houseplants", "plant",
        "plants", "flowerpot", "flowerpots", "flower pot", "indoor plant",
        "indoor plants", "potted flower", "succulent", "succulents", "bonsai",
        "potted greenery", "desk plant", "fern", "cactus", "flora"
    ],
    "bed": [
        "bed", "beds", "mattress", "mattresses", "bunk bed", "cot", "cots",
        "futon bed", "twin bed", "queen bed", "king bed", "double bed", "headboard"
    ],
    "dining table": [
        "dining table", "dining tables", "dinner table", "kitchen table",
        "table", "tables", "desk", "desks", "dining-table", "dining desk",
        "coffee table", "workstation", "picnic table", "study desk", "study table",
        "counter", "countertop", "workbench"
    ],
    "toilet": [
        "toilet", "toilets", "commode", "commodes", "restroom", "wc",
        "lavatory", "water closet", "urinal", "latrine"
    ],
    "tv": [
        "tv", "tvs", "television", "televisions", "flat screen", "flatscreen",
        "monitor", "monitors", "screen", "screens", "telly", "display", "displays",
        "smart tv", "oled", "led tv", "tv screen", "computer screen", "tv monitor"
    ],
    "laptop": [
        "laptop", "laptops", "notebook", "notebooks", "computer", "computers",
        "pc", "macbook", "chromebook", "thinkpad", "portable computer",
        "laptop computer", "workstation", "laptop pc"
    ],
    "mouse": [
        "mouse", "mice", "computer mouse", "trackball", "optical mouse",
        "wireless mouse", "pc mouse"
    ],
    "remote": [
        "remote", "remotes", "remote control", "remote controls", "tv remote",
        "clicker", "clickers", "controller", "controllers"
    ],
    "keyboard": [
        "keyboard", "keyboards", "computer keyboard", "key board",
        "typing keyboard", "mechanical keyboard", "wireless keyboard"
    ],
    "cell phone": [
        "cell phone", "cell phones", "cellphone", "cellphones", "phone", "phones", "mobile",
        "mobiles", "smartphone", "smartphones", "smart phone", "smart phones", "iphone", "iphones",
        "android", "androids", "mobile phone", "telephone", "telephones",
        "cellular", "cell", "handset", "device"
    ],
    "microwave": [
        "microwave", "microwaves", "microwave oven"
    ],
    "oven": [
        "oven", "ovens", "stove", "stoves", "stovetop", "stovetops",
        "range", "cooktop", "baking oven", "kitchen oven"
    ],
    "toaster": [
        "toaster", "toasters", "toaster oven"
    ],
    "sink": [
        "sink", "sinks", "washbasin", "washbasins", "basin", "basins",
        "kitchen sink", "bathroom sink", "vanity sink", "handbasin"
    ],
    "refrigerator": [
        "refrigerator", "refrigerators", "fridge", "fridges", "freezer",
        "freezers", "icebox", "cooler"
    ],
    "book": [
        "book", "books", "novel", "novels", "textbook", "textbooks",
        "booklet", "paperback", "hardback", "hardcover", "reading book",
        "magazine", "volume", "tome"
    ],
    "clock": [
        "clock", "clocks", "wall clock", "alarm clock", "timepiece",
        "timepieces", "digital clock", "analog clock", "timer"
    ],
    "vase": [
        "vase", "vases", "flower vase", "urn", "flower holder"
    ],
    "scissors": [
        "scissors", "scissor", "pair of scissors", "shears", "clippers",
        "trimmer", "cutting scissors"
    ],
    "teddy bear": [
        "teddy bear", "teddy bears", "teddy", "teddies", "plushie",
        "plushies", "stuffed bear", "stuffed animal", "soft toy"
    ],
    "hair drier": [
        "hair drier", "hair driers", "hair dryer", "hair dryers",
        "blowdryer", "blowdryers", "blow dryer", "blow dryers", "hairdryer"
    ],
    "toothbrush": [
        "toothbrush", "toothbrushes", "electric toothbrush"
    ],
}

# Invert alias map for fast lookup: alias -> class_name
ALIAS_TO_CLASS: dict[str, str] = {}
for cls_name, aliases in CLASS_ALIASES.items():
    for alias in aliases:
        ALIAS_TO_CLASS[alias.lower().strip()] = cls_name

# Sort aliases by word length descending so longest matches take precedence
# (e.g. "hot dog" matched before "dog", "teddy bear" before "bear", "baseball bat" before "baseball")
ALL_ALIASES_SORTED: list[str] = sorted(
    ALIAS_TO_CLASS.keys(),
    key=lambda s: (len(s.split()), len(s)),
    reverse=True,
)

# ---------------------------------------------------------------------------
# Grammatical Rules & Lemmatization Dictionaries
# ---------------------------------------------------------------------------
IRREGULAR_PLURALS: dict[str, str] = {
    "people": "person",
    "men": "man",
    "women": "woman",
    "children": "child",
    "kids": "kid",
    "mice": "mouse",
    "knives": "knife",
    "calves": "calf",
    "oxen": "ox",
    "geese": "goose",
    "feet": "foot",
    "teeth": "tooth",
    "puppies": "puppy",
    "kitties": "kitty",
    "ponies": "pony",
    "cacti": "cactus",
}

# Negation indicators within a local context window (1-3 tokens prior)
NEGATION_WORDS: set[str] = {
    "not", "no", "never", "except", "without", "excluding", "avoid",
    "don't", "dont", "neither", "nor", "skip", "ignore", "exclude"
}

# Conjunctions or reset words that end an existing negation clause
NEGATION_RESET_WORDS: set[str] = {
    "but", "instead", "rather", "however", "although", "while", "plus", "and"
}

# Numbers & quantities mapping
WORD_NUMBERS: dict[str, int] = {
    "a": 1, "an": 1, "one": 1, "single": 1,
    "two": 2, "pair": 2, "couple": 2,
    "three": 3, "few": 3,
    "four": 4, "several": 4,
    "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10,
    "dozen": 12
}

# Common object attributes / adjectives / materials
COLOR_WORDS: set[str] = {
    "red", "green", "blue", "yellow", "black", "white", "orange", "purple",
    "pink", "brown", "gray", "grey", "silver", "gold", "golden", "dark", "light"
}

MATERIAL_WORDS: set[str] = {
    "wooden", "plastic", "metal", "glass", "paper", "ceramic", "leather", "cotton"
}

SIZE_WORDS: set[str] = {
    "big", "large", "small", "tiny", "huge", "tall", "short", "long", "little"
}

SPATIAL_WORDS: set[str] = {
    "left", "right", "center", "middle", "front", "back", "top", "bottom",
    "near", "far", "here", "there"
}

STOP_BOUNDARY_WORDS: set[str] = {
    "and", "or", "with", "near", "on", "in", "at", "by", "under", "over",
    "from", "to", "behind", "next"
}


class NLPExtractor:
    """
    Robust NLP Object Extractor tailored for speech-to-text outputs.
    Extracts objects mapped to the 80 COCO classes.
    """

    def __init__(self):
        self.coco_classes = COCO_CLASSES
        self.class_to_id = CLASS_NAME_TO_ID
        self.alias_to_class = ALIAS_TO_CLASS
        self.sorted_aliases = ALL_ALIASES_SORTED
        self.phonetic_map = PHONETIC_NORMALIZATION

    def get_class_id(self, name_or_alias: str) -> int | None:
        """Resolve a class name, alias, or lemma to its canonical COCO ID (0-79)."""
        key = name_or_alias.strip().lower()
        if key in self.class_to_id:
            return self.class_to_id[key]
        if key in self.alias_to_class:
            canonical = self.alias_to_class[key]
            return self.class_to_id.get(canonical)
        lemma = self.lemmatize_word(key)
        if lemma in self.class_to_id:
            return self.class_to_id[lemma]
        if lemma in self.alias_to_class:
            canonical = self.alias_to_class[lemma]
            return self.class_to_id.get(canonical)
        return None

    def id_to_class(self, class_id: int) -> str | None:
        """Return canonical class name for COCO ID (0-79)."""
        if 0 <= class_id < len(self.coco_classes):
            return self.coco_classes[class_id]
        return None

    def normalize_sentence(self, sentence: str) -> str:
        """Clean and normalize spoken sentence contractions, sound-alikes, and punctuation."""
        s = sentence.lower().strip()

        # Remove common assistant wake phrases when they appear in the query
        wake_patterns = [
            r"^(?:hi|hey|hello|ok|okay)?\s*(?:omii|omi|omee)\b\s*[:,]?",
            r"\b(?:hi|hey|hello|ok|okay)\s+(?:omii|omi|omee)\b",
            r"\b(?:omii|omi|omee)\b",
        ]
        for wp in wake_patterns:
            s = re.sub(wp, " ", s, flags=re.IGNORECASE)

        # Normalize dotted acronyms (e.g. t.v. -> tv, p.c. -> pc)
        s = re.sub(r"\bt\.v\.?\b", "tv", s)
        s = re.sub(r"\bp\.c\.?\b", "pc", s)

        # Expand common speech contractions
        contractions = [
            (r"\bcan't\b", "cannot"),
            (r"\bwon't\b", "will not"),
            (r"\bdon't\b", "do not"),
            (r"\bdid'nt\b", "did not"),
            (r"\bdoesn't\b", "does not"),
            (r"\bwhere's\b", "where is"),
            (r"\bthere's\b", "there is"),
            (r"\bwhat's\b", "what is"),
            (r"\bi'm\b", "i am"),
            (r"\bi'd\b", "i would"),
            (r"\bit's\b", "it is"),
            (r"\blet's\b", "let us"),
        ]
        for pattern, replacement in contractions:
            s = re.sub(pattern, replacement, s)

        # Remove filler hesitations common in STT output
        fillers = [r"\bumm?\b", r"\buhh?\b", r"\bahh?\b", r"\berr?\b", r"\bhmm?\b"]
        for f in fillers:
            s = re.sub(f, "", s)

        # Normalize hyphenated compound words (e.g. cell-phone -> cell phone, wine-glass -> wine glass)
        s = re.sub(r"(\w+)-(\w+)", r"\1 \2", s)

        # Apply phonetic and sound-alike STT normalizations
        for misspelling, correction in self.phonetic_map.items():
            pattern = r"\b" + re.escape(misspelling) + r"\b"
            s = re.sub(pattern, correction, s)

        # Standardize whitespace and remove unnecessary punctuation
        s = re.sub(r"[?!.,;:\"']", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def lemmatize_word(self, word: str) -> str:
        """Reduce a plural or inflected word to its singular lemma."""
        word = word.lower()
        if word in IRREGULAR_PLURALS:
            return IRREGULAR_PLURALS[word]

        # Inherent words ending in s
        if word in {"skis", "scissors", "glasses", "bus", "gas", "citrus"}:
            return word

        if word.endswith("ies") and len(word) > 4:
            return word[:-3] + "y"
        if word.endswith("es") and len(word) > 3:
            for suffix in ("shes", "ches", "sses", "xes", "zes"):
                if word.endswith(suffix):
                    return word[:-2]
            if word.endswith("ves") and len(word) > 4:
                return word[:-3] + "fe"
        if word.endswith("s") and not word.endswith("ss") and len(word) > 2:
            return word[:-1]

        return word

    def detect_intent(self, sentence: str) -> str:
        """Classify user conversational intent."""
        s = f" {sentence.lower()} "
        if any(trigger in s for trigger in [
            "where is", "where are", "where", "locate", "find", "track",
            "spot", "search for", "look for", "show me where", "pointing to"
        ]):
            return "locate"
        if any(trigger in s for trigger in [
            "how many", "count", "number of", "calculate count", "total of", "sum of"
        ]):
            return "count"
        if any(trigger in s for trigger in [
            "only show", "filter by", "highlight only", "focus on", "just show", "keep only"
        ]):
            return "filter"
        if any(trigger in s for trigger in [
            "is there", "are there", "do you see", "can you see", "detect",
            "identify", "recognize", "check for", "look at"
        ]):
            return "detect"
        return "general"

    def is_negated(self, sentence: str, start_char: int) -> bool:
        """
        Check if an object mentioned at start_char is preceded by a negation.
        Respects contrastive boundary tokens ('not X, but Y').
        """
        prefix = sentence[:start_char].strip()
        prefix_tokens = prefix.split()
        if not prefix_tokens:
            return False

        # Look back up to 6 words
        window = [t.lower().strip("?!.,;:\"'") for t in prefix_tokens[-6:]]
        # Iterate backwards from closest token
        for token in reversed(window):
            if token in NEGATION_RESET_WORDS:
                # If we encounter a reset conjunction before hitting negation, it's not negated
                return False
            if token in NEGATION_WORDS:
                return True
        return False

    def extract_attributes_and_count(
        self, sentence: str, start_char: int
    ) -> tuple[int | None, list[str]]:
        """
        Extract nearby quantifier/count and descriptive attributes for an object.
        Scans backwards towards the immediate modifier boundary.
        """
        prefix = sentence[:start_char].strip()
        tokens = prefix.split()
        if not tokens:
            return None, []

        count: int | None = None
        attributes: list[str] = []

        # Scan backwards up to 3 tokens before this object
        rev_window = [t.lower().strip("?!.,;:\"'") for t in reversed(tokens[-3:])]

        for token in rev_window:
            # If we hit a boundary word like 'and' or 'or', stop looking further back for numbers/colors
            if token in STOP_BOUNDARY_WORDS:
                break

            if count is None:
                if token in WORD_NUMBERS:
                    count = WORD_NUMBERS[token]
                elif token.isdigit():
                    try:
                        count = int(token)
                    except ValueError:
                        pass

            if token in COLOR_WORDS:
                attributes.append(token)
            elif token in MATERIAL_WORDS:
                attributes.append(token)
            elif token in SIZE_WORDS:
                attributes.append(token)
            elif token in SPATIAL_WORDS:
                attributes.append(token)

        attributes.reverse()
        return count, attributes

    def fuzzy_match(self, word: str, threshold: float = 0.82) -> tuple[str, float] | None:
        """Fuzzy match against 80 class names and aliases for STT typos."""
        w_len = len(word)
        if w_len < 3:
            return None

        # Short words (len 3): check common phonetic typos directly
        if w_len == 3:
            short_typos = {
                "cub": "cup", "cop": "cup", "cap": "cup", "keg": "cup",
                "kar": "car", "carr": "car",
                "beg": "backpack", "bak": "backpack",
                "bet": "bed",
                "dok": "dog", "dogg": "dog",
                "kat": "cat",
                "buss": "bus",
                "tye": "tie",
                "kow": "cow",
            }
            if word in short_typos:
                return short_typos[word], 0.90
            return None

        # Dynamically adjust threshold based on word length to prevent short false matches
        req_threshold = 0.86 if w_len <= 5 else threshold

        best_match = None
        best_score = 0.0

        # Match against canonical COCO class names first
        for cls_name in self.coco_classes:
            score = difflib.SequenceMatcher(None, word, cls_name).ratio()
            if score > best_score:
                best_score = score
                best_match = cls_name

        if best_match and best_score >= req_threshold:
            return best_match, round(best_score, 3)

        # Match against aliases (requiring first letter match and >=0.88 to prevent false positives like slippers -> clippers)
        for alias, cls_name in self.alias_to_class.items():
            if " " not in alias and len(alias) >= 4:
                if word[0] != alias[0]:
                    continue
                score = difflib.SequenceMatcher(None, word, alias).ratio()
                if score > best_score:
                    best_score = score
                    best_match = cls_name

        if best_match and best_score >= 0.88:
            return best_match, round(best_score, 3)

        return None

    def extract(
        self,
        sentence: str,
        include_attributes: bool = True,
        allow_fuzzy: bool = True,
        filter_negated: bool = True,
    ) -> NLPExtractionResponse:
        """
        Main NLP Object Extraction pipeline.

        Args:
            sentence: Raw text from speech-to-text or user query
            include_attributes: Whether to capture counts, colors, sizes
            allow_fuzzy: Whether to allow fuzzy matching for STT phonetic/spelling noise
            filter_negated: Whether to exclude negated objects from target_classes

        Returns:
            NLPExtractionResponse with structured metadata and YOLO-ready target IDs
        """
        start_time = time.perf_counter()
        cleaned = self.normalize_sentence(sentence)
        intent = self.detect_intent(sentence)

        extracted_objects: list[ExtractedObject] = []
        occupied_spans: list[tuple[int, int]] = []

        def is_overlapping(start: int, end: int) -> bool:
            for s, e in occupied_spans:
                if max(start, s) < min(end, e):
                    return True
            return False

        # -------------------------------------------------------------------
        # Step 1: Multi-word & Exact Alias Matching (Longest Match First)
        # -------------------------------------------------------------------
        for alias in self.sorted_aliases:
            pattern = r"\b" + re.escape(alias) + r"\b"
            for m in re.finditer(pattern, cleaned, flags=re.IGNORECASE):
                start, end = m.start(), m.end()
                if not is_overlapping(start, end):
                    cls_name = self.alias_to_class[alias]
                    cls_id = self.class_to_id[cls_name]
                    is_exact = alias == cls_name
                    match_type: Literal["exact", "alias", "lemma", "fuzzy"] = (
                        "exact" if is_exact else "alias"
                    )
                    confidence = 1.0 if is_exact else 0.95

                    negated = self.is_negated(cleaned, start)
                    count, attrs = (
                        self.extract_attributes_and_count(cleaned, start)
                        if include_attributes
                        else (None, [])
                    )

                    extracted_objects.append(
                        ExtractedObject(
                            class_id=cls_id,
                            class_name=cls_name,
                            matched_phrase=alias,
                            match_type=match_type,
                            confidence=confidence,
                            negated=negated,
                            count=count,
                            attributes=attrs,
                            start_char=start,
                            end_char=end,
                        )
                    )
                    occupied_spans.append((start, end))

        # -------------------------------------------------------------------
        # Step 2: Lemmatized Matching on remaining single tokens
        # -------------------------------------------------------------------
        words = re.finditer(r"\b[a-zA-Z0-9_\-]+\b", cleaned)
        for m in words:
            start, end = m.start(), m.end()
            if is_overlapping(start, end):
                continue

            raw_word = m.group(0).lower()
            lemma = self.lemmatize_word(raw_word)

            if lemma in self.alias_to_class:
                cls_name = self.alias_to_class[lemma]
                cls_id = self.class_to_id[cls_name]
                negated = self.is_negated(cleaned, start)
                count, attrs = (
                    self.extract_attributes_and_count(cleaned, start)
                    if include_attributes
                    else (None, [])
                )

                extracted_objects.append(
                    ExtractedObject(
                        class_id=cls_id,
                        class_name=cls_name,
                        matched_phrase=raw_word,
                        match_type="lemma",
                        confidence=0.92,
                        negated=negated,
                        count=count,
                        attributes=attrs,
                        start_char=start,
                        end_char=end,
                    )
                )
                occupied_spans.append((start, end))

            # ---------------------------------------------------------------
            # Step 3: Fuzzy matching fallback (difflib) for STT errors
            # ---------------------------------------------------------------
            elif allow_fuzzy and len(raw_word) >= 3:
                fuzzy_res = self.fuzzy_match(raw_word)
                if fuzzy_res:
                    cls_name, sim_score = fuzzy_res
                    cls_id = self.class_to_id[cls_name]
                    negated = self.is_negated(cleaned, start)
                    count, attrs = (
                        self.extract_attributes_and_count(cleaned, start)
                        if include_attributes
                        else (None, [])
                    )

                    extracted_objects.append(
                        ExtractedObject(
                            class_id=cls_id,
                            class_name=cls_name,
                            matched_phrase=raw_word,
                            match_type="fuzzy",
                            confidence=sim_score,
                            negated=negated,
                            count=count,
                            attributes=attrs,
                            start_char=start,
                            end_char=end,
                        )
                    )
                    occupied_spans.append((start, end))

        # Sort extracted objects by their appearance order in the sentence
        extracted_objects.sort(key=lambda obj: obj.start_char)

        # Deduplicate and separate target classes vs excluded classes
        target_classes_set: list[str] = []
        target_class_ids_set: list[int] = []
        excluded_classes_set: list[str] = []

        for obj in extracted_objects:
            if obj.negated:
                if obj.class_name not in excluded_classes_set:
                    excluded_classes_set.append(obj.class_name)
            else:
                if obj.class_name not in target_classes_set:
                    target_classes_set.append(obj.class_name)
                    target_class_ids_set.append(obj.class_id)

        # If filter_negated is active, remove excluded from target
        if filter_negated:
            filtered_targets: list[str] = []
            filtered_ids: list[int] = []
            for name, cid in zip(target_classes_set, target_class_ids_set):
                if name not in excluded_classes_set:
                    filtered_targets.append(name)
                    filtered_ids.append(cid)
            target_classes_set = filtered_targets
            target_class_ids_set = filtered_ids

        elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 3)

        return NLPExtractionResponse(
            success=True,
            original_sentence=sentence,
            cleaned_sentence=cleaned,
            intent=intent,
            target_classes=target_classes_set,
            target_class_ids=target_class_ids_set,
            excluded_classes=excluded_classes_set,
            extracted_objects=extracted_objects,
            processing_time_ms=elapsed_ms,
        )

    def get_supported_classes(self) -> SupportedClassesResponse:
        """Return the directory of all 80 COCO classes with alias details and examples."""
        details: list[ClassDetail] = []
        for idx, cls_name in enumerate(self.coco_classes):
            aliases = CLASS_ALIASES.get(cls_name, [cls_name])
            examples = [
                f"Where is the {cls_name}?",
                f"Can you find my {aliases[1] if len(aliases) > 1 else cls_name}?",
                f"Detect two {aliases[0]}s near the table",
            ]
            details.append(
                ClassDetail(
                    class_id=idx,
                    class_name=cls_name,
                    aliases=aliases[:10],
                    example_prompts=examples,
                )
            )
        return SupportedClassesResponse(total_classes=len(details), classes=details)

    def extract_requested_region(self, sentence: str) -> str | None:
        """
        Extracts requested spatial direction / region from speech or text.
        Returns one of:
          'top-left', 'top-right', 'bottom-left', 'bottom-right',
          'left', 'right', 'top', 'bottom', 'center'
        or None if no spatial constraint was specified.
        """
        text = sentence.lower()

        # Check compound 2D directions first (e.g. top-left, bottom-right)
        if re.search(r"\b(top[- ]left|upper[- ]left)\b", text):
            return "top-left"
        if re.search(r"\b(top[- ]right|upper[- ]right)\b", text):
            return "top-right"
        if re.search(r"\b(bottom[- ]left|lower[- ]left)\b", text):
            return "bottom-left"
        if re.search(r"\b(bottom[- ]right|lower[- ]right)\b", text):
            return "bottom-right"

        # Center / middle
        if re.search(r"\b(center|middle|in the center|in the middle|centered)\b", text):
            return "center"

        # Explicit left phrases
        if re.search(r"\b(on the left|to the left|at the left|towards the left|in the left|left[- ]side|left side|left corner|left half|left portion|left hand side|on left)\b", text):
            return "left"

        # Explicit right phrases
        if re.search(r"\b(on the right|to the right|at the right|towards the right|in the right|right[- ]side|right side|right corner|right half|right portion|right hand side|on right)\b", text):
            return "right"

        # Explicit top phrases
        if re.search(r"\b(on top|at the top|to the top|in the top|top side|top half|top portion|upper side|upper part|upper corner)\b", text):
            return "top"

        # Explicit bottom phrases
        if re.search(r"\b(at the bottom|on the bottom|to the bottom|in the bottom|bottom side|bottom half|bottom portion|lower side|lower part)\b", text):
            return "bottom"

        # Standalone 'left' if not part of non-directional idioms ('i left', 'is left', 'are left', 'left over', 'left behind')
        if re.search(r"\bleft\b", text) and not re.search(r"\b(i left|we left|was left|is left|are left|left over|left behind|have left)\b", text):
            return "left"

        # Standalone 'right'
        if re.search(r"\bright\b", text) and not re.search(r"\b(all right|right now|right away|that's right|thats right)\b", text):
            return "right"

        # Standalone 'top' or 'bottom'
        if re.search(r"\btop\b", text) and not re.search(r"\b(laptop|tabletop|desktop)\b", text):
            return "top"
        if re.search(r"\bbottom\b", text):
            return "bottom"

        return None

    def extract_target_and_intent(self, transcribed_text: str) -> dict[str, Any]:
        """
        High-level helper for interactive mode:
        Extracts primary target object, class ID, conversational intent,
        and requested spatial region from transcribed speech.
        """
        extraction = self.extract(transcribed_text)
        target_name = extraction.target_classes[0] if extraction.target_classes else None
        target_id = extraction.target_class_ids[0] if extraction.target_class_ids else None
        region = self.extract_requested_region(transcribed_text)

        intent = extraction.intent
        if region and intent in ("locate", "detect"):
            intent = "locate_in_region"

        return {
            "target_object": target_name,
            "target_class_id": target_id,
            "target_classes": extraction.target_classes,
            "target_class_ids": extraction.target_class_ids,
            "intent": intent,
            "requested_region": region,
            "original_query": transcribed_text,
            "valid": bool(target_name is not None),
            "processing_time_ms": extraction.processing_time_ms,
        }



_DEFAULT_EXTRACTOR: NLPExtractor | None = None


def get_nlp_extractor() -> NLPExtractor:
    """Singleton getter for the NLPExtractor."""
    global _DEFAULT_EXTRACTOR
    if _DEFAULT_EXTRACTOR is None:
        _DEFAULT_EXTRACTOR = NLPExtractor()
    return _DEFAULT_EXTRACTOR
