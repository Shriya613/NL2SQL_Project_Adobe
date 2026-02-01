BEGIN TRANSACTION;
CREATE TABLE Content_Curating_Techniques(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    technique TEXT);
INSERT INTO "Content_Curating_Techniques" VALUES(1,'Algorithmic_Recommendations');
INSERT INTO "Content_Curating_Techniques" VALUES(2,'Editorial_Picks');
INSERT INTO "Content_Curating_Techniques" VALUES(3,'Collaborative_Filtering');
INSERT INTO "Content_Curating_Techniques" VALUES(4,'Trending_Topics');
INSERT INTO "Content_Curating_Techniques" VALUES(5,'User_Feedback');
CREATE TABLE Streaming_Services(
    service_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    genre TEXT);
INSERT INTO "Streaming_Services" VALUES(1,'Netflix','Movies/TV Shows');
INSERT INTO "Streaming_Services" VALUES(2,'Amazon Prime Video','Movies/TV Shows');
INSERT INTO "Streaming_Services" VALUES(3,'Hulu','TV Shows/Movies');
INSERT INTO "Streaming_Services" VALUES(4,'Disney+','Family/Kids');
INSERT INTO "Streaming_Services" VALUES(5,'HBO Max','Movies/TV Shows');
CREATE TABLE Streaming_Services_Content_Curating_Techniques(
    streaming_services_content_curating_technique_id INTEGER PRIMARY KEY AUTOINCREMENT,
    streaming_service_id INTEGER,
    content_curating_technique_id INTEGER,
    FOREIGN KEY (streaming_service_id) REFERENCES Streaming_Services(service_id),
    FOREIGN KEY (content_curating_technique_id) REFERENCES Content_Curating_Techniques(id));
INSERT INTO "Streaming_Services_Content_Curating_Techniques" VALUES(1,1,1);
INSERT INTO "Streaming_Services_Content_Curating_Techniques" VALUES(2,1,2);
INSERT INTO "Streaming_Services_Content_Curating_Techniques" VALUES(3,2,3);
INSERT INTO "Streaming_Services_Content_Curating_Techniques" VALUES(4,3,4);
INSERT INTO "Streaming_Services_Content_Curating_Techniques" VALUES(5,4,5);
INSERT INTO "Streaming_Services_Content_Curating_Techniques" VALUES(6,2,1);
INSERT INTO "Streaming_Services_Content_Curating_Techniques" VALUES(7,3,1);
INSERT INTO "Streaming_Services_Content_Curating_Techniques" VALUES(8,4,1);
INSERT INTO "Streaming_Services_Content_Curating_Techniques" VALUES(9,5,1);
CREATE TABLE Users(
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    password TEXT);
INSERT INTO "Users" VALUES(1,'JohnDoe123','password123');
INSERT INTO "Users" VALUES(2,'JaneSmith456','securepass456');
INSERT INTO "Users" VALUES(3,'AliceBrown789','myPassword789');
INSERT INTO "Users" VALUES(4,'CharlieBlack012','strongPass012');
INSERT INTO "Users" VALUES(5,'EveWhite345','easyPeasy345');
CREATE TABLE Views(
    view_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    streamed_service_id INTEGER,
    viewed_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES Users(user_id),
    FOREIGN KEY (streamed_service_id) REFERENCES Streaming_Services(service_id));
INSERT INTO "Views" VALUES(1,1,1,'2022-01-01 10:00:00');
INSERT INTO "Views" VALUES(2,2,2,'2022-01-02 15:30:00');
INSERT INTO "Views" VALUES(3,3,3,'2022-01-03 20:45:00');
INSERT INTO "Views" VALUES(4,4,4,'2022-01-04 23:50:00');
INSERT INTO "Views" VALUES(5,5,5,'2022-01-05 00:10:00');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Content_Curating_Techniques',5);
INSERT INTO "sqlite_sequence" VALUES('Streaming_Services',5);
INSERT INTO "sqlite_sequence" VALUES('Users',5);
INSERT INTO "sqlite_sequence" VALUES('Views',5);
INSERT INTO "sqlite_sequence" VALUES('Streaming_Services_Content_Curating_Techniques',9);
COMMIT;
