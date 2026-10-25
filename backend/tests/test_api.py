from conftest import create_project_and_dataset, login, make_png


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["db"] is True


def test_openapi_schema_builds(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert "/api/v1/datasets/{dataset_id}/uploads" in paths
    assert "/api/v1/review/queue" in paths


def test_requires_login(client):
    assert client.get("/api/v1/projects").status_code == 401


def test_wrong_password_rejected(client):
    login(client, "admin")
    r = client.post("/api/v1/auth/login", json={"email": "admin@test.local", "password": "nope"})
    assert r.status_code == 401


def test_admin_creates_and_lists_projects(admin_client):
    r = admin_client.post("/api/v1/projects", json={"name": "Osa demo"})
    assert r.status_code == 201
    assert [p["name"] for p in admin_client.get("/api/v1/projects").json()] == ["Osa demo"]
    assert admin_client.get("/api/v1/auth/me").json()["role"] == "admin"


def test_viewer_cannot_create_projects_or_review(client):
    login(client, "viewer")
    assert client.post("/api/v1/projects", json={"name": "x"}).status_code == 403
    assert client.get("/api/v1/review/queue", params={"project_id": 1}).status_code == 403


def test_upload_validates_content_and_deduplicates(admin_client):
    _, dataset_id = create_project_and_dataset(admin_client)
    png = make_png()
    files = [
        ("files", ("a.png", png, "image/png")),
        ("files", ("notes.jpg", b"this is not an image", "image/jpeg")),  # extension lies
        ("files", ("a-copy.png", png, "image/png")),
        ("files", ("empty.png", b"", "image/png")),
    ]
    body = admin_client.post(f"/api/v1/datasets/{dataset_id}/uploads", files=files).json()
    codes = [(r["filename"], r["ok"], r["code"]) for r in body["results"]]
    assert codes == [
        ("a.png", True, None),
        ("notes.jpg", False, "UNSUPPORTED_TYPE"),
        ("a-copy.png", False, "DUPLICATE"),
        ("empty.png", False, "EMPTY_FILE"),
    ]
    assert body["job_id"] is not None
    media = admin_client.get("/api/v1/media", params={"dataset_id": dataset_id}).json()
    assert len(media) == 1 and media[0]["status"] == "uploaded"
    assert admin_client.get(f"/api/v1/media/{media[0]['id']}/file").content == png


def test_aerial_video_is_rejected_explicitly(admin_client):
    _, dataset_id = create_project_and_dataset(admin_client, modality="aerial")
    mp4 = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64
    body = admin_client.post(f"/api/v1/datasets/{dataset_id}/uploads",
                             files=[("files", ("flight.mp4", mp4, "video/mp4"))]).json()
    assert body["results"][0]["code"] == "UNSUPPORTED_MODALITY"
    assert body["job_id"] is None
