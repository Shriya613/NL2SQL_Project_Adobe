BEGIN TRANSACTION;
CREATE TABLE "Consulted_Sources" (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    specialty TEXT
);
INSERT INTO "Consulted_Sources" VALUES(1,'John Smith','Politics');
INSERT INTO "Consulted_Sources" VALUES(2,'Jane Doe','Economy');
INSERT INTO "Consulted_Sources" VALUES(3,'Bob Johnson','Environmental Science');
INSERT INTO "Consulted_Sources" VALUES(4,'Alice Brown','Technology');
INSERT INTO "Consulted_Sources" VALUES(5,'Michael White','Healthcare');
CREATE TABLE "Documentary" (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    release_date DATE,
    EXPERT_ID INTEGER,
    FOREIGN KEY(EXPERT_ID) REFERENCES "Consulted_Sources"(id)
);
INSERT INTO "Documentary" VALUES(1,'The Rise of Political Polarization','2020-09-01',1);
INSERT INTO "Documentary" VALUES(2,'The Future of Work','2021-07-15',2);
INSERT INTO "Documentary" VALUES(3,'Green New Deal Explained','2020-10-20',3);
INSERT INTO "Documentary" VALUES(4,'AI Ethics & Society','2021-02-01',4);
INSERT INTO "Documentary" VALUES(5,'Global Pandemic Response','2020-12-10',5);
CREATE TABLE "Interviews" (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TIMESTAMP,
    details TEXT,
    EXPERT_ID INTEGER,
    FOREIGN KEY(EXPERT_ID) REFERENCES "Consulted_Sources"(id)
);
INSERT INTO "Interviews" VALUES(1,'2021-07-25 10:00:00','Interview about economic recovery post-pandemic.',2);
INSERT INTO "Interviews" VALUES(2,'2021-08-10 14:30:00','Discussion on environmental policies and their impact.',3);
INSERT INTO "Interviews" VALUES(3,'2021-09-05 11:00:00','Insights on future technology trends and challenges.',4);
INSERT INTO "Interviews" VALUES(4,'2021-10-15 16:00:00','Analysis of healthcare system improvements needed.',5);
INSERT INTO "Interviews" VALUES(5,'2021-11-01 10:30:00','Opinions on political polarization and its effects.',1);
CREATE TABLE "News_Articles" (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    headline TEXT,
    content TEXT,
    publication_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    DOCUMENTARY_ID INTEGER,
    EXPERT_ID INTEGER,
    FOREIGN KEY(DOCUMENTARY_ID) REFERENCES Documentary(id),
    FOREIGN KEY(EXPERT_ID) REFERENCES "Consulted_Sources"(id)
);
INSERT INTO "News_Articles" VALUES(1,'US Midterm Elections Approach','Discusses key issues and candidates...','2024-02-19 14:43:33',1,1);
INSERT INTO "News_Articles" VALUES(2,'Automation Impact on Jobs','Explores potential solutions...','2024-02-19 14:43:33',2,2);
INSERT INTO "News_Articles" VALUES(3,'Deforestation Crisis Worsens','Highlights recent developments...','2024-02-19 14:43:33',3,3);
INSERT INTO "News_Articles" VALUES(4,'China''s AI Advancements','Analyzes China''s progress in AI...','2024-02-19 14:43:33',4,4);
INSERT INTO "News_Articles" VALUES(5,'COVID Vaccine Distribution','Updates on global vaccine distribution...','2024-02-19 14:43:33',5,5);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Consulted_Sources',5);
INSERT INTO "sqlite_sequence" VALUES('Documentary',5);
INSERT INTO "sqlite_sequence" VALUES('News_Articles',5);
INSERT INTO "sqlite_sequence" VALUES('Interviews',5);
COMMIT;
