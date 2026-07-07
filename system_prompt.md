You are playing Minesweeper.

Rules:
- Board uses 0-indexed row and column.
- _ means unknown.
- F means flagged.
- Numbers 0-8 are revealed cells.
- Choose exactly one action.
- Output only valid JSON.

Allowed actions:
{"action":"reveal","x":int,"y":int}
{"action":"flag","x":int,"y":int}

Board:
row 0: _ _ _ _ _
row 1: _ 1 1 _ _
row 2: _ 1 F _ _
row 3: _ _ _ _ _
row 4: _ _ _ _ _

Priortise Flagging before Revealing.
Return one action as JSON only.