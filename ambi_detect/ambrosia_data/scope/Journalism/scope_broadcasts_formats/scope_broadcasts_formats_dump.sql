BEGIN TRANSACTION;
CREATE TABLE Articles(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    content TEXT,
    publish_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    author_id INTEGER,
    FOREIGN KEY(author_id) REFERENCES Authors(id));
INSERT INTO "Articles" VALUES(1,'The Importance of Journalistic Integrity','Journalism plays a critical role in society...','2024-01-30 18:51:09',1);
INSERT INTO "Articles" VALUES(2,'Investigative Reporting Techniques','A look at some effective techniques used by investigative reporters...','2024-01-30 18:51:09',2);
INSERT INTO "Articles" VALUES(3,'Impact of Social Media on News Consumption','How social media has changed the way people consume news...','2024-01-30 18:51:09',3);
INSERT INTO "Articles" VALUES(4,'Censorship and Freedom of the Press','Examining the issue of censorship and its impact on freedom of the press...','2024-01-30 18:51:09',4);
INSERT INTO "Articles" VALUES(5,'The Future of Print Journalism','Can print journalism survive in the digital age?','2024-01-30 18:51:09',5);
CREATE TABLE Authors(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    email TEXT);
INSERT INTO "Authors" VALUES(1,'John Doe','john@example.com');
INSERT INTO "Authors" VALUES(2,'Jane Smith','jane@example.com');
INSERT INTO "Authors" VALUES(3,'Mike Johnson','mike@example.com');
INSERT INTO "Authors" VALUES(4,'Emily Davis','emily@example.com');
INSERT INTO "Authors" VALUES(5,'David Brown','david@example.com');
CREATE TABLE Broadcasts(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    date DATETIME);
INSERT INTO "Broadcasts" VALUES(1,'Tonight''s Live News Coverage','2023-04-01 18:00:00');
INSERT INTO "Broadcasts" VALUES(2,'Weekly Podcast: Current Affairs','2023-04-06 09:00:00');
INSERT INTO "Broadcasts" VALUES(3,'Daily Online Newspaper Update','2023-04-07 12:00:00');
INSERT INTO "Broadcasts" VALUES(4,'Monthly Video Magazine Issue','2023-04-15 17:00:00');
INSERT INTO "Broadcasts" VALUES(5,'Breaking News Special','2023-04-20 15:00:00');
CREATE TABLE Broadcasts_Formats(
    broadcast_id INTEGER,
    format_id INTEGER,
    FOREIGN KEY(broadcast_id) REFERENCES Broadcasts(id),
    FOREIGN KEY(format_id) REFERENCES Formats(id));
INSERT INTO "Broadcasts_Formats" VALUES(1,5);
INSERT INTO "Broadcasts_Formats" VALUES(1,2);
INSERT INTO "Broadcasts_Formats" VALUES(2,3);
INSERT INTO "Broadcasts_Formats" VALUES(3,2);
INSERT INTO "Broadcasts_Formats" VALUES(4,4);
INSERT INTO "Broadcasts_Formats" VALUES(5,5);
INSERT INTO "Broadcasts_Formats" VALUES(2,5);
INSERT INTO "Broadcasts_Formats" VALUES(3,5);
INSERT INTO "Broadcasts_Formats" VALUES(4,5);
CREATE TABLE Comments(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    comment TEXT,
    article_id INTEGER,
    FOREIGN KEY(article_id) REFERENCES Articles(id));
INSERT INTO "Comments" VALUES(1,'Great article! Very informative.',1);
INSERT INTO "Comments" VALUES(2,'I agree completely - it''s essential that journalists maintain their integrity.',1);
INSERT INTO "Comments" VALUES(3,'This is an interesting perspective on how social media affects news consumption.',3);
INSERT INTO "Comments" VALUES(4,'Important topic, but I think there should also be a discussion about self-censorship.',4);
INSERT INTO "Comments" VALUES(5,'Print journalism will always have a place as long as there is a demand for quality reporting.',5);
CREATE TABLE Formats(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    format_name TEXT);
INSERT INTO "Formats" VALUES(1,'Live');
INSERT INTO "Formats" VALUES(2,'Online');
INSERT INTO "Formats" VALUES(3,'Podcast');
INSERT INTO "Formats" VALUES(4,'Video');
INSERT INTO "Formats" VALUES(5,'Magazine');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('Authors',5);
INSERT INTO "sqlite_sequence" VALUES('Articles',5);
INSERT INTO "sqlite_sequence" VALUES('Comments',5);
INSERT INTO "sqlite_sequence" VALUES('Formats',5);
INSERT INTO "sqlite_sequence" VALUES('Broadcasts',5);
COMMIT;
