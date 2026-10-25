"""Write 220 NEW held-out sentences (no overlap with spike sentences.json or ref.json).

Sources: prose, news, dialogue, numbers/dates/currency.
Output: heldout.json in the repo (branch g2p-v1).
"""
import json
import os
import unicodedata

REPO = r"C:\Users\bansa\OneDrive\Desktop\TTS-8M-model"
SPIKE = r"C:\Users\bansa\OneDrive\Desktop\g2p-spike"

prose = [
    "The old lighthouse keeper counted the waves until sleep finally came.",
    "Every morning she walked to the bakery on the corner and bought a loaf of rye bread.",
    "The garden had gone wild over the summer, but the tomatoes kept coming.",
    "He fixed the leaking faucet with a wrench he inherited from his grandfather.",
    "The library was quiet except for the sound of pages turning somewhere in the back.",
    "A cold wind swept down from the hills and rattled the shutters all night.",
    "The children built a fort out of cardboard boxes in the living room.",
    "She read the letter twice, folded it carefully, and put it back in the drawer.",
    "The train arrived three minutes late, which was unusual for a Tuesday.",
    "Rain fell steadily on the tin roof, making conversation almost impossible.",
    "The dog waited by the door every afternoon until the school bus came.",
    "He stirred the soup slowly and added a pinch of salt, then tasted it again.",
    "The mountain trail was steep, but the view from the top was worth every step.",
    "They spent the whole evening arguing about which movie to watch.",
    "The old clock in the hallway chimed ten times before it finally stopped.",
    "Her handwriting was small and neat, the kind that fits twice as much on a page.",
    "The fishermen set out before dawn, when the harbor was still wrapped in fog.",
    "A single lamp burned in the window of the cottage across the field.",
    "The recipe called for two cups of flour, a spoonful of sugar, and patience.",
    "He polished his boots, straightened his collar, and walked out into the snow.",
    "The museum guide explained that the painting was older than the building itself.",
    "Bees moved lazily between the lavender bushes in the warm afternoon light.",
    "The letter lay unopened on the table for three days before she found the courage.",
    "Wolves had not been seen in the valley for decades, until last winter.",
    "The bridge across the river was built in a single summer by local stonemasons.",
    "She kept a jar of buttons on the shelf, collected from shirts worn out long ago.",
    "The orchestra tuned their instruments while the audience settled into their seats.",
    "A thin layer of ice covered the pond, and the ducks were not pleased about it.",
    "The farmer checked the fence line every evening, even in the pouring rain.",
    "His grandfather's stories always began the same way, with a knock on the door.",
    "The city skyline glowed orange as the sun dropped behind the water tower.",
    "Nobody remembered who had planted the apple tree, but everyone picked from it.",
    "The cat stretched, yawned, and knocked a pen off the desk on purpose.",
    "They followed the map until the road turned to gravel and then to nothing.",
    "The smell of fresh coffee drifted up the stairs and pulled him out of bed.",
    "She won the race by half a stride, arms raised before she crossed the line.",
    "The attic was full of boxes labeled in faded ink from three generations back.",
    "A woodpecker hammered at the dead oak outside his window every morning.",
    "The ferry left the dock at six, whether the fog lifted or not.",
    "He learned to play chess from a neighbor who never once let him win.",
    "The quilt on her bed was stitched from dresses she wore as a girl.",
    "Storm clouds gathered over the ridge long before the first drop fell.",
    "The bakery sold out of cinnamon rolls before eight most Saturdays.",
    "He taped the map to the wall and traced the route with a red marker.",
    "The twins shared everything, including the blame and the last cookie.",
    "An old violin sat in its case in the corner, waiting for someone to ask.",
    "The path to the spring was overgrown, but the water was still cold and clear.",
    "She counted the stairs out of habit, seventeen from the street to her door.",
    "The campfire crackled and threw sparks into a sky crowded with stars.",
    "By the time the kettle whistled, the argument had already been forgotten.",
    "The tailor measured him twice, frowned, and started over from the shoulder.",
    "Frost covered the windshield, and the scraper was buried somewhere in the trunk.",
    "The village well had been dry for years, but nobody had the heart to fill it in.",
    "He whistled while he worked, badly, and the dog howled along in protest.",
    "The last bus of the night rattled past the shuttered shops on Main Street.",
    "Her garden notes filled three notebooks, with sketches of every bed and border.",
    "The boys raced their bikes down the hill and coasted all the way to the bridge.",
    "A kettle of hawks circled above the harvested fields in the late October sky.",
    "The piano had not been tuned in years, but she played it anyway.",
    "He kept his tools arranged by size, and woe to anyone who moved a wrench.",
    "The night market opened at dusk, and the smell of grilled corn took over the street.",
    "She patched the tent fly with tape and prayed the weather would cooperate.",
    "The retired teacher volunteered at the shelter every Wednesday without fail.",
    "Snow squeaked under their boots as they climbed the short hill to the bus stop.",
    "The radio played softly in the garage while he rebuilt the carburetor.",
    "A row of sunflowers along the fence leaned east, following the morning light.",
    "The town's one traffic light blinked yellow long after midnight.",
    "She wrote the grocery list on the back of an old concert ticket.",
    "The puppy hid a sock under the couch and pretended nothing had happened.",
    "Morning fog pooled in the valley until the sun burned it away around nine.",
    "He signed the card, sealed the envelope, and realized he had no stamps.",
    "The choir practiced in the church basement, voices rising through the floorboards.",
    "Her bicycle had one working gear and a bell that scared the neighbors.",
    "The porch swing creaked in rhythm with the conversation on summer nights.",
]

news = [
    "The city council approved a budget of $4.7 million for road repairs on Monday.",
    "Officials said the new water treatment plant will open in March 2026.",
    "The storm knocked out power to about 12,000 homes across the county.",
    "Shares of the company rose 8.3% after the earnings report was released.",
    "A jury deliberated for six hours before reaching a verdict on Friday.",
    "The transit authority announced fare increases starting January 15.",
    "Firefighters contained the wildfire at roughly 2,400 acres by nightfall.",
    "The school board voted 6 to 1 to extend the school year by one week.",
    "Unemployment in the region fell to 3.9%, the lowest level in a decade.",
    "The bridge closure will add about 20 minutes to the morning commute.",
    "Police said the accident happened shortly after 11 p.m. on Route 9.",
    "The museum will host a free admission day on August 24.",
    "Residents have until December 31 to submit comments on the zoning plan.",
    "The hospital reported 214 new patients during the first week of flu season.",
    "A record 45,000 runners registered for this year's marathon.",
    "The library system received a $250,000 grant to expand its digital archive.",
    "Crews repaired the broken water main in under 5 hours, officials said.",
    "The airline added 3 new routes from the regional airport this spring.",
    "Voters will decide on a $60 million school bond in November.",
    "The bakery chain recalled about 1,800 loaves over a labeling error.",
    "Temperatures are expected to drop to 17 degrees by Thursday night.",
    "The port handled 2.1 million containers in the third quarter.",
    "A new bike lane will connect downtown to the riverfront by June.",
    "The council rejected the proposal in a 4 to 3 vote after two hours of debate.",
    "The utility restored electricity to the final 300 customers around 9:30 a.m.",
    "The festival drew an estimated 75,000 visitors over the weekend.",
    "Inspectors found no violations during the surprise audit on Tuesday.",
    "The county sheriff's office is investigating a string of break-ins on Elm Street.",
    "The university announced a $15 million donation for a new science hall.",
    "Train service between the two cities will be suspended on Sunday.",
    "The farmers' market will move indoors starting October 12.",
    "A water advisory was lifted Wednesday after tests came back clean.",
    "The company plans to hire 150 workers at its plant outside Toledo.",
    "The highway project, delayed twice, is now set to finish in 2027.",
    "Local honey production jumped 22% this year, beekeepers reported.",
    "The city's oldest bookstore closed its doors after 84 years in business.",
    "Officials urged residents to conserve water through the end of August.",
    "The school district will hire 40 new teachers before the fall term.",
    "A section of the trail will be closed for repairs from May 6 to May 20.",
    "The fire department rescued three hikers stranded on the north ridge.",
]

dialogue = [
    "\"Did you lock the door?\" she asked, already halfway down the stairs.",
    "\"I'll be there in five minutes,\" he said, still tying his shoes.",
    "\"That's not what I meant,\" she said quietly, \"and you know it.\"",
    "\"Can we stop for coffee?\" \"Only if it's quick,\" her father replied.",
    "\"You're not going to believe what happened at the store today,\" she began.",
    "\"Put the box down slowly,\" the mover said, \"it's heavier than it looks.\"",
    "\"Whose turn is it to walk the dog?\" \"Not mine,\" said both children at once.",
    "\"I thought the meeting was at 2:30,\" he said, checking his watch.",
    "\"It's on the counter, next to the bread,\" she called from the other room.",
    "\"We should leave before 6,\" he warned, \"traffic gets ugly after that.\"",
    "\"Do you really think it will snow?\" asked the boy at the window.",
    "\"That's the third time this week the printer has jammed,\" she sighed.",
    "\"Grandma's recipe calls for a whole stick of butter,\" he admitted.",
    "\"I'll take the early train,\" she said, \"you pick me up at the station.\"",
    "\"Don't touch that dial,\" the mechanic said, wiping his hands on a rag.",
    "\"How much do I owe you?\" \"Twelve dollars even,\" the driver answered.",
    "\"The doctor said ten days of rest,\" she groaned, \"ten whole days.\"",
    "\"It wasn't me,\" said the cat, in the way that cats say such things.",
    "\"Are we nearly there yet?\" came the inevitable question from the back seat.",
    "\"Read it again,\" the child demanded, \"the part with the dragon.\"",
    "\"You can't be serious,\" he laughed, but she was already packing.",
    "\"The number you have dialed has been disconnected,\" the recording repeated.",
    "\"I'll have the soup and a side of bread,\" she told the waiter.",
    "\"Meet me at the fountain at noon,\" the note read, \"and come alone.\"",
    "\"That's the last time I trust the weather forecast,\" he muttered.",
    "\"Your table will be ready in about 15 minutes,\" the hostess said.",
    "\"It's not the heat, it's the humidity,\" everyone in the office agreed.",
    "\"I never said that,\" she protested, \"check the email yourself.\"",
    "\"One ticket to the late show, please,\" he said, counting out the change.",
    "\"Happy birthday!\" they shouted as the lights came on.",
]

numbers = [
    "The meeting starts at 9:15 sharp, so don't be late.",
    "She was born on July 4, 1990, in a small town in Ohio.",
    "The invoice totaled $1,284.67, including tax and shipping.",
    "He read 342 pages of the novel in a single weekend.",
    "The temperature climbed to 98 degrees by mid-afternoon.",
    "Their phone number ends in 0042, which she found easy to remember.",
    "The project has a budget of $2.5 million and a deadline of October 1.",
    "About 3,500 people attended the open-air concert on Saturday.",
    "The train to Boston leaves at 6:45 a.m. and arrives before 10.",
    "He paid $19.99 for the adapter and felt robbed anyway.",
    "The census counted 48,213 residents in the township last year.",
    "Her flight lands at 11:20 p.m., long after the last bus.",
    "The recipe serves 6 and takes about 45 minutes to prepare.",
    "They drove 1,200 miles in three days and swore never again.",
    "The company was founded in 1987 by two college roommates.",
    "Tickets cost $35 for adults and $12 for children under 10.",
    "The warehouse stores over 90,000 units at peak season.",
    "It was 3:00 in the morning when the phone finally rang.",
    "The marathon record stands at 2 hours, 1 minute, and 39 seconds.",
    "She saved $50 a week for a year to buy the used piano.",
    "The library extended its hours until 9 p.m. on weekdays.",
    "His great-grandfather was born in 1901 and lived to be 97.",
    "The download finished in 4 minutes and 12 seconds.",
    "A round-trip ticket costs $148 during the off-season.",
    "The exam has 60 questions and a time limit of 90 minutes.",
    "The charity raised $86,000 during its annual telethon.",
    "Bus 27 stops at the corner every 20 minutes or so.",
    "The building has 14 floors and exactly one working elevator.",
    "He guessed the jar held 730 marbles; the real count was 712.",
    "The conference runs from March 3 to March 7 this year.",
    "Her apartment is on the 5th floor, walk-up, no air conditioning.",
    "The bill came to $43.20, and they split it three ways.",
    "The bridge is 2,780 feet long and opened to traffic in 1962.",
    "Sales grew from $1.1 million to $3.4 million in two years.",
    "The store closes at 8 p.m. on Sundays and 10 the rest of the week.",
    "He celebrated his 40th birthday with a hike and a very large cake.",
    "The report listed 12 recommendations, but only 3 were adopted.",
    "Water pressure dropped around 7:30 a.m. in the eastern neighborhoods.",
    "The auction's opening bid was $500 for the antique desk.",
    "She set 6 alarms and slept through every one of them.",
]

mixed = [
    "On February 14, the florist sold out of roses by 10 a.m.",
    "The 2nd edition of the handbook came out in 2019 and sold 30,000 copies.",
    "He deposited $500 on Friday and withdrew $80 on Monday.",
    "The 3rd act of the play runs about 40 minutes without an intermission.",
    "By 2025, the town expects to finish its fiber network at a cost of $9 million.",
    "The 1st of April is the only day the office allows pranks.",
    "She paid $12.50 for the 4th-row seat and called it a bargain.",
    "The 22nd annual film festival opens on September 9.",
    "Between 8 and 9 inches of rain fell in a single night.",
    "The 100th customer of the day won a $25 gift card.",
    "At 5:45 the bakery's lights were already on, and the ovens were hot.",
    "The 19th-century mill was converted into 24 apartments in 2011.",
    "He donated 15 inches of his hair to the charity in June.",
    "The 10th caller won concert tickets worth about $180.",
    "On December 24 the office closes at noon, and the janitor locks up.",
    "The 7:10 ferry is the one the commuters take; the 7:40 is for tourists.",
    "She ran her 1st marathon at 52 and her 5th at 58.",
    "The invoice dated June 30 was paid in full on July 28.",
    "The 9th inning lasted 34 minutes and included two home runs.",
    "Rent is due on the 1st, and the late fee is $35 after the 5th.",
]

all_sents = [
    *[(t, "prose") for t in prose],
    *[(t, "news") for t in news],
    *[(t, "dialogue") for t in dialogue],
    *[(t, "numbers") for t in numbers],
    *[(t, "mixed") for t in mixed],
]

# --- overlap check (case-insensitive, whitespace-normalized) against spike files ---
def norm(s):
    return " ".join(s.lower().split())

spike_texts = set()
for fn in ("sentences.json", "ref.json"):
    data = json.load(open(os.path.join(SPIKE, fn), encoding="utf-8"))
    for d in data:
        spike_texts.add(norm(d["text"]))

seen = set()
out = []
for i, (text, source) in enumerate(all_sents):
    n = norm(text)
    assert n not in spike_texts, f"OVERLAP with spike: {text!r}"
    assert n not in seen, f"DUPLICATE: {text!r}"
    seen.add(n)
    out.append({"id": i, "source": source, "text": text})

assert len(out) >= 200
path = os.path.join(REPO, "heldout.json")
with open(path, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(f"wrote {len(out)} sentences to {path}")
print("overlap check: PASS (0 matches, case-insensitive, whitespace-normalized)")
