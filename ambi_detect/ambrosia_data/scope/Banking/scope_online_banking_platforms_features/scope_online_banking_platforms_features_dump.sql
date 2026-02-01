BEGIN TRANSACTION;
CREATE TABLE Features(
    feature_id INTEGER PRIMARY KEY AUTOINCREMENT,
    feature_value VARCHAR(256));
INSERT INTO "Features" VALUES(1,'Bill Pay');
INSERT INTO "Features" VALUES(2,'Mobile App');
INSERT INTO "Features" VALUES(3,'International Transfers');
INSERT INTO "Features" VALUES(4,'Secure Login');
INSERT INTO "Features" VALUES(5,'Account Statements');
CREATE TABLE Online_Banking_Platforms(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    website TEXT);
INSERT INTO "Online_Banking_Platforms" VALUES(1,'Platform1','www.bank1.com');
INSERT INTO "Online_Banking_Platforms" VALUES(2,'Platform2','www.bank2.com');
INSERT INTO "Online_Banking_Platforms" VALUES(3,'Platform3','www.bank3.com');
INSERT INTO "Online_Banking_Platforms" VALUES(4,'Platform4','www.bank4.com');
INSERT INTO "Online_Banking_Platforms" VALUES(5,'Platform5','www.bank5.com');
CREATE TABLE Online_Banking_Platforms_Features(
    platform_id INTEGER,
    feature_id INTEGER,
    FOREIGN KEY(platform_id) REFERENCES Online_Banking_Platforms(id),
    FOREIGN KEY(feature_id) REFERENCES Features(feature_id));
INSERT INTO "Online_Banking_Platforms_Features" VALUES(1,1);
INSERT INTO "Online_Banking_Platforms_Features" VALUES(1,2);
INSERT INTO "Online_Banking_Platforms_Features" VALUES(2,1);
INSERT INTO "Online_Banking_Platforms_Features" VALUES(2,3);
INSERT INTO "Online_Banking_Platforms_Features" VALUES(3,1);
INSERT INTO "Online_Banking_Platforms_Features" VALUES(3,2);
INSERT INTO "Online_Banking_Platforms_Features" VALUES(4,1);
INSERT INTO "Online_Banking_Platforms_Features" VALUES(4,3);
INSERT INTO "Online_Banking_Platforms_Features" VALUES(5,1);
INSERT INTO "Online_Banking_Platforms_Features" VALUES(5,2);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Online_Banking_Platforms',5);
INSERT INTO "sqlite_sequence" VALUES('Features',5);
COMMIT;
