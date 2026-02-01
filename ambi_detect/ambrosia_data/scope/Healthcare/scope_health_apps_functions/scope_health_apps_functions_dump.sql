BEGIN TRANSACTION;
CREATE TABLE Functions(
    functionID INTEGER PRIMARY KEY AUTOINCREMENT,
    value TEXT);
INSERT INTO "Functions" VALUES(1,'Activity_Tracking');
INSERT INTO "Functions" VALUES(2,'Calorie_Counting');
INSERT INTO "Functions" VALUES(3,'Meditation');
INSERT INTO "Functions" VALUES(4,'Disease_Symptoms');
INSERT INTO "Functions" VALUES(5,'Blood_Pressure_Monitoring');
CREATE TABLE Health_Apps(
    appID INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    description TEXT,
    price REAL );
INSERT INTO "Health_Apps" VALUES(1,'App1','This is App1',0.99);
INSERT INTO "Health_Apps" VALUES(2,'App2','This is App2',1.99);
INSERT INTO "Health_Apps" VALUES(3,'App3','This is App3',2.99);
INSERT INTO "Health_Apps" VALUES(4,'App4','This is App4',3.99);
INSERT INTO "Health_Apps" VALUES(5,'App5','This is App5',4.99);
CREATE TABLE Health_Apps_Functions(
    healthAppID INTEGER,
    functionID INTEGER,
    FOREIGN KEY(healthAppID) REFERENCES Health_Apps(appID),
    FOREIGN KEY(functionID) REFERENCES Functions(functionID),
    PRIMARY KEY(healthAppID, functionID));
INSERT INTO "Health_Apps_Functions" VALUES(1,1);
INSERT INTO "Health_Apps_Functions" VALUES(2,1);
INSERT INTO "Health_Apps_Functions" VALUES(3,1);
INSERT INTO "Health_Apps_Functions" VALUES(4,1);
INSERT INTO "Health_Apps_Functions" VALUES(5,1);
INSERT INTO "Health_Apps_Functions" VALUES(5,3);
CREATE TABLE Reviews(
    reviewID INTEGER PRIMARY KEY AUTOINCREMENT,
    rating INTEGER ,
    comment TEXT,
    appID INTEGER,
    userID INTEGER,
    FOREIGN KEY(appID) REFERENCES Health_Apps(appID),
    FOREIGN KEY(userID) REFERENCES Users(userID));
INSERT INTO "Reviews" VALUES(1,5,'Great app!',1,1);
INSERT INTO "Reviews" VALUES(2,4,'Good but could use some improvements.',2,2);
INSERT INTO "Reviews" VALUES(3,3,'Worked well overall.',3,3);
INSERT INTO "Reviews" VALUES(4,2,'Not what I expected.',4,4);
INSERT INTO "Reviews" VALUES(5,1,'Didn''t work as promised.',5,5);
CREATE TABLE Users(
    userID INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    email TEXT,
    password TEXT);
INSERT INTO "Users" VALUES(1,'User1','user1@example.com','password1');
INSERT INTO "Users" VALUES(2,'User2','user2@example.com','password2');
INSERT INTO "Users" VALUES(3,'User3','user3@example.com','password3');
INSERT INTO "Users" VALUES(4,'User4','user4@example.com','password4');
INSERT INTO "Users" VALUES(5,'User5','user5@example.com','password5');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Users',5);
INSERT INTO "sqlite_sequence" VALUES('Health_Apps',5);
INSERT INTO "sqlite_sequence" VALUES('Functions',5);
INSERT INTO "sqlite_sequence" VALUES('Reviews',5);
COMMIT;
