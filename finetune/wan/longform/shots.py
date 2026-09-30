"""Shot lists for the long-form remakes, written from the client's storyboards (iteration_1/30_prompts)
in the v2 skit-caption style the golden was trained on (processed/v1v2_75clip/metadata.json)."""
STYLE = ("A 2D cartoon animation in the Pudgy Penguins style, with thick clean black outlines, soft cel "
         "shading and flat pastel colors, showing ")
PAX = "Pax, a short round blue penguin with a white face and belly and a small orange beak"
POLLY = "Polly, a short round pink penguin with a white face and belly, rosy cheeks and a small orange beak"
CAM = "; the camera stays still, a fixed vertical frame."
K = "/workspace/longform/keyframes"

LIFE_WITH_HER = [
  dict(id="lwh_1", key=f"{K}/lwh_1_sandwich.png", frames=81,
       prompt=STYLE + PAX + ", sitting on a wooden chair at a round cream dining table, happily eating a sandwich "
       "from a white plate; a pink flipper reaches in from the right edge of the frame and takes the other "
       "sandwich half off his plate, and Pax smiles warmly as he looks off to the right" + CAM),
  dict(id="lwh_2", key=f"{K}/lwh_2_popcorn.png", frames=73,
       prompt=STYLE + PAX + ", sitting on a cream couch holding a striped popcorn bucket and eating popcorn; a pink "
       "flipper reaches in from the side and grabs a handful of popcorn out of the bucket, and Pax looks to his "
       "left and smiles" + CAM),
  dict(id="lwh_3", key=f"{K}/lwh_3_icecream.png", frames=65,
       prompt=STYLE + PAX + ", sitting on a pink park bench outdoors under a blue sky licking a green ice cream "
       "cone; a pink flipper reaches in and swaps his cone for a pink one, and Pax chuckles happily" + CAM),
  dict(id="lwh_4", key=f"{K}/lwh_4_fries.png", frames=73,
       prompt=STYLE + PAX + ", at a bright turquoise table eating french fries from a red and white striped cup "
       "beside a paper bag; a pink flipper reaches in and grabs some fries, and Pax looks up surprised and then "
       "smiles" + CAM),
  dict(id="lwh_5", key=f"{K}/lwh_5_together.png", frames=57,
       prompt=STYLE + PAX + " and " + POLLY + ", standing side by side at a turquoise table sharing french fries; "
       "Polly giggles and leans in to give Pax a kiss on the cheek and he blushes happily" + CAM),
]

EATING_STAGES = [
  dict(id="eat_1", key=f"{K}/eat_1_week.png", end=f"{K}/eat_1_end.png", frames=65,
       prompt=STYLE + PAX + " and " + POLLY + ", sitting across from each other at a small round cream table eating "
       "noodles from two dark green bowls; Polly takes the tiniest dainty bite and covers her beak shyly with her "
       "flipper as she chews, while Pax slurps his noodles happily; the two dark green noodle bowls stay on the "
       "table the whole time" + CAM),
  dict(id="eat_2", key=f"{K}/eat_2_month.png", end=f"{K}/eat_2_end.png", frames=73,
       prompt=STYLE + PAX + " and " + POLLY + ", sitting at a small round cream table with trays of sushi; Polly "
       "slowly picks up a single piece of sushi from her own tray with her flipper, carries it across the table and "
       "gently places it onto Pax's tray in one smooth continuous motion, and Pax smiles at her; both sushi trays and "
       "the small soy sauce dish stay on the table the whole time" + CAM),
  dict(id="eat_3", key=f"{K}/eat_3_rolls.png", end=f"{K}/eat_3_end.png", frames=81,
       prompt=STYLE + PAX + " and " + POLLY + ", sitting at a small round cream table, each holding a big seaweed-wrapped "
       "rice roll in their flippers and eating it; Polly quickly gobbles hers down, finishes and sits back contentedly, "
       "while Pax keeps calmly eating his roll" + CAM),
  dict(id="eat_4", key=f"{K}/eat_4_closeup.png", end=f"{K}/eat_4_end.png", frames=49,
       prompt=STYLE + POLLY + ", sitting at a round cream table tipping a big bowl up over her face and drinking "
       "every last drop, then lowering the bowl to reveal crumbs and sauce all over her face" + CAM),
  dict(id="eat_5", key=f"{K}/eat_5_end.png", frames=49,
       prompt=STYLE + PAX + " and " + POLLY + ", sitting at a small round cream table with empty bowls; Polly, with "
       "crumbs on her face, leans back with a satisfied deep sigh while Pax stares at her with his beak open" + CAM),
]
