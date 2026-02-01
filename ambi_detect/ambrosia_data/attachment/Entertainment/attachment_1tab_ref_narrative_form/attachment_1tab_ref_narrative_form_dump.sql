BEGIN TRANSACTION;
CREATE TABLE Authors(
    id INTEGER PRIMARY KEY,
    Name TEXT,
    Birthdate DATE,
    Deathdate DATE
);
INSERT INTO "Authors" VALUES(1,'Author A','1874-03-16',NULL);
INSERT INTO "Authors" VALUES(2,'Author B','1925-08-15','1989-05-25');
INSERT INTO "Authors" VALUES(3,'Author C','1930-12-10','2005-03-07');
CREATE TABLE Epochs(
    id INTEGER PRIMARY KEY,
    Epoch TEXT
);
INSERT INTO "Epochs" VALUES(1,'20th Century');
INSERT INTO "Epochs" VALUES(2,'19th Century');
CREATE TABLE NarrativeForms(
    id INTEGER PRIMARY KEY,
    Narrative_Form TEXT
);
INSERT INTO "NarrativeForms" VALUES(1,'Autobiography');
INSERT INTO "NarrativeForms" VALUES(2,'Historical Account');
CREATE TABLE "Narratives"(
    id INTEGER PRIMARY KEY,
    narrative_form_id INTEGER,
    epoch_id INTEGER,
    title TEXT,
    publication_year INT, `author_id` INTEGER REFERENCES `Authors`(`id`),
    FOREIGN KEY (narrative_form_id) REFERENCES NarrativeForms(id),
    FOREIGN KEY (epoch_id) REFERENCES Epochs(id)
);
INSERT INTO "Narratives" VALUES(1,1,1,'My Life Story',2000,1);
INSERT INTO "Narratives" VALUES(2,1,2,'More of My Life',2005,2);
INSERT INTO "Narratives" VALUES(3,2,1,'Events That Shaped The World',1997,1);
INSERT INTO "Narratives" VALUES(4,2,2,'Another Historical Event',2003,3);
INSERT INTO "Narratives" VALUES(5,3,3,'Random Thoughts',2008,1);
INSERT INTO "Narratives" VALUES(6,4,4,'How I Overcame Fear',2012,3);
INSERT INTO "Narratives" VALUES(7,5,5,'Life Lessons Learned',2011,1);
COMMIT;
