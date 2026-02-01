BEGIN TRANSACTION;
CREATE TABLE assigned_stories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    story_id INTEGER,
    journalist_id INTEGER,
    start_date DATE,
    end_date DATE,
    FOREIGN KEY (story_id) REFERENCES stories(id),
    FOREIGN KEY (journalist_id) REFERENCES journalists(id)
);
INSERT INTO "assigned_stories" VALUES(1,1,1,'2022-09-01','2022-09-15');
INSERT INTO "assigned_stories" VALUES(2,2,2,'2022-09-02','2022-09-16');
CREATE TABLE journalists (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    email TEXT,
    phone VARCHAR(15)
);
INSERT INTO "journalists" VALUES(1,'John Doe','johndoe@example.com','+1234567890');
INSERT INTO "journalists" VALUES(2,'Jane Smith','janesmith@example.com','+0987654321');
CREATE TABLE news_agencies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    location TEXT,
    contact_email TEXT UNIQUE
);
INSERT INTO "news_agencies" VALUES(1,'Agencia Noticias','Madrid, Spain','contact@noticias.es');
INSERT INTO "news_agencies" VALUES(2,'News Hub','New York City, USA','contact@newshub.us');
CREATE TABLE "stories" (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    content TEXT,
    publish_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    Reporter INTEGER,
    FOREIGN KEY (Reporter) REFERENCES journalists(id));
INSERT INTO "stories" VALUES(1,'US Election Results','This is an article about recent election results.','2024-02-19 15:13:19',1);
INSERT INTO "stories" VALUES(2,'European Economy Overview','An overview of European economies.','2024-02-19 15:13:19',2);
CREATE TABLE story_categories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    category_name TEXT,
    description TEXT
);
INSERT INTO "story_categories" VALUES(1,'Politics','Stories about government and political activities.');
INSERT INTO "story_categories" VALUES(2,'Economy','Articles covering financial matters and economic trends.');
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('journalists',2);
INSERT INTO "sqlite_sequence" VALUES('news_agencies',2);
INSERT INTO "sqlite_sequence" VALUES('story_categories',2);
INSERT INTO "sqlite_sequence" VALUES('assigned_stories',2);
INSERT INTO "sqlite_sequence" VALUES('stories',2);
COMMIT;
