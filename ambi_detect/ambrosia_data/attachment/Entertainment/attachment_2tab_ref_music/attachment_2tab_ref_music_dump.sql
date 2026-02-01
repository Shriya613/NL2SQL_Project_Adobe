BEGIN TRANSACTION;
CREATE TABLE Album(
    id INTEGER PRIMARY KEY,
    Title TEXT,
    Year INT,
    Singer_id INTEGER,
    FOREIGN KEY (Singer_id) REFERENCES Singer(id));
INSERT INTO "Album" VALUES(1,'Avalon Blues',1967,1);
INSERT INTO "Album" VALUES(2,'Biograph',1980,2);
INSERT INTO "Album" VALUES(3,'Electric Mud',1968,3);
INSERT INTO "Album" VALUES(4,'The Complete Studio Recordings',1991,4);
INSERT INTO "Album" VALUES(5,'Bluegrass',1969,5);
CREATE TABLE Band_Member(
    id INTEGER PRIMARY KEY,
    Name TEXT,
    Instrument_id INTEGER,
    FOREIGN KEY (Instrument_id) REFERENCES Instrument(id));
INSERT INTO "Band_Member" VALUES(1,'Eddie Boyd',1);
INSERT INTO "Band_Member" VALUES(2,'Big Bill Broonzy',2);
INSERT INTO "Band_Member" VALUES(3,'Sunnyland Slim',3);
INSERT INTO "Band_Member" VALUES(4,'Memphis Minnie',4);
INSERT INTO "Band_Member" VALUES(5,'Jimmy Rogers',5);
CREATE TABLE Blues_Song(
    id INTEGER PRIMARY KEY,
    Song_Name TEXT,
    Genre TEXT,
    Singer_id INTEGER,
    FOREIGN KEY (Singer_id) REFERENCES Singer(id));
INSERT INTO "Blues_Song" VALUES(1,'I Got A Mind To Give Up The Blues','Blues',1);
INSERT INTO "Blues_Song" VALUES(2,'Cross Road Blues','Blues',3);
INSERT INTO "Blues_Song" VALUES(3,'Hoochie Coochie Man','Blues',2);
INSERT INTO "Blues_Song" VALUES(4,'St. Louis Blues','Blues',4);
INSERT INTO "Blues_Song" VALUES(5,'Broken Hearted Blues','Blues',1);
CREATE TABLE Folk_Song(
    id INTEGER PRIMARY KEY,
    Song_Name TEXT,
    Genre TEXT,
    Singer_id INTEGER,
    FOREIGN KEY (Singer_id) REFERENCES Singer(id));
INSERT INTO "Folk_Song" VALUES(1,'Man Of Constant Sorrow','Folk',1);
INSERT INTO "Folk_Song" VALUES(2,'Tom Dooley','Folk',3);
INSERT INTO "Folk_Song" VALUES(3,'The House Carpenter','Folk',5);
INSERT INTO "Folk_Song" VALUES(4,'Cripple Creek Ferry','Folk',4);
INSERT INTO "Folk_Song" VALUES(5,'Pretty Polly','Folk',2);
CREATE TABLE Instrument(
    id INTEGER PRIMARY KEY,
    Type TEXT);
INSERT INTO "Instrument" VALUES(1,'Guitar');
INSERT INTO "Instrument" VALUES(2,'Piano');
INSERT INTO "Instrument" VALUES(3,'Drums');
INSERT INTO "Instrument" VALUES(4,'Violin');
INSERT INTO "Instrument" VALUES(5,'Harmonica');
CREATE TABLE Singer(
    id INTEGER PRIMARY KEY,
    Name TEXT);
INSERT INTO "Singer" VALUES(1,'John Hurt');
INSERT INTO "Singer" VALUES(2,'Robert Johnson');
INSERT INTO "Singer" VALUES(3,'Muddy Waters');
INSERT INTO "Singer" VALUES(4,'Bessie Smith');
INSERT INTO "Singer" VALUES(5,'Bill Monroe');
COMMIT;
