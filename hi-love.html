<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>A small promise</title>

  <!-- Google Fonts -->
  <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Inter:wght@300;400;600&family=Dancing+Script:wght@400;600&display=swap" rel="stylesheet">

  <style>
    :root{
      --bg1: #f7f6fb;
      --bg2: #fffaf6;
      --accent: #c28bff;
      --accent-2: #ffb4c1;
      --muted: #667085;
      --card: #ffffff;
      --glass: rgba(255,255,255,0.65);
      --shadow: 0 10px 30px rgba(24, 26, 33, 0.08);
      --radius: 18px;
    }

    html,body{
      height:100%;
      margin:0;
      font-family: "Inter", system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
      background: linear-gradient(180deg, var(--bg1) 0%, #fff 40%), radial-gradient(800px 400px at 10% 10%, rgba(194,139,255,0.08), transparent 10%), radial-gradient(600px 300px at 90% 90%, rgba(255,180,193,0.06), transparent 10%);
      -webkit-font-smoothing:antialiased;
      -moz-osx-font-smoothing:grayscale;
      color:#102a43;
    }

    .wrap{
      min-height:100vh;
      display:flex;
      align-items:center;
      justify-content:center;
      padding:40px 24px;
      box-sizing:border-box;
    }

    .card{
      width:100%;
      max-width:720px;
      background: linear-gradient(180deg, rgba(255,255,255,0.9), var(--card));
      border-radius:calc(var(--radius) + 6px);
      box-shadow: var(--shadow);
      padding:36px;
      position:relative;
      overflow:hidden;
      border:1px solid rgba(16, 42, 67, 0.04);
      backdrop-filter: blur(6px);
    }

    /* decorative corner glow */
    .card::before{
      content:"";
      position:absolute;
      width:420px;
      height:420px;
      right:-120px;
      top:-120px;
      background: radial-gradient(circle at 30% 30%, rgba(194,139,255,0.16), transparent 25%), radial-gradient(circle at 70% 70%, rgba(255,180,193,0.12), transparent 30%);
      transform: rotate(12deg);
      pointer-events:none;
    }

    header{
      display:flex;
      align-items:center;
      gap:16px;
      margin-bottom:18px;
    }

    .badge{
      width:62px;
      height:62px;
      border-radius:14px;
      display:flex;
      align-items:center;
      justify-content:center;
      background: linear-gradient(135deg, var(--accent), var(--accent-2));
      color:white;
      font-weight:700;
      font-family:"Playfair Display", serif;
      font-size:20px;
      box-shadow: 0 6px 20px rgba(194,139,255,0.12);
    }

    h1{
      margin:0;
      font-family:"Playfair Display", serif;
      font-weight:600;
      font-size:20px;
      color:#081325;
    }
    p.lead{
      margin:0;
      font-size:13px;
      color:var(--muted);
      margin-top:4px;
    }

    .message{
      margin-top:18px;
      padding:22px;
      border-radius:14px;
      background: linear-gradient(180deg, rgba(249,248,252,0.9), rgba(255,255,255,0.85));
      border: 1px dashed rgba(16,42,67,0.04);
      font-size:16px;
      line-height:1.55;
      color:#13293b;
    }

    .message .quote{
      display:block;
      font-family:"Playfair Display", serif;
      font-size:19px;
      color:#09121a;
      margin-bottom:10px;
    }

    .message .soft{
      color: #334e68;
      font-size:15px;
    }

    .signature{
      margin-top:18px;
      display:flex;
      align-items:center;
      gap:12px;
    }

    .heart{
      width:56px;
      height:56px;
      border-radius:12px;
      display:flex;
      align-items:center;
      justify-content:center;
      background: linear-gradient(180deg, #ffe7f0, #ffdfe7);
      box-shadow: 0 8px 20px rgba(255,180,193,0.18);
      font-family:"Dancing Script", cursive;
      font-size:22px;
      color:#b0406e;
    }

    .byline{
      font-size:15px;
      color:#0b2440;
    }
    .byline small{
      display:block;
      font-size:12px;
      color:var(--muted);
      margin-top:6px;
    }

    footer{
      margin-top:22px;
      display:flex;
      gap:10px;
      align-items:center;
      justify-content:space-between;
      flex-wrap:wrap;
    }

    .btn{
      display:inline-flex;
      align-items:center;
      gap:10px;
      background: linear-gradient(90deg, #8a63ff, #ff8fb0);
      color:white;
      padding:10px 14px;
      border-radius:12px;
      text-decoration:none;
      font-weight:600;
      box-shadow: 0 8px 28px rgba(138,99,255,0.14);
      transition:transform .18s ease, box-shadow .18s ease;
      font-size:14px;
    }
    .btn:hover{ transform:translateY(-3px); box-shadow: 0 18px 40px rgba(138,99,255,0.16); }

    .muted{
      color:var(--muted); font-size:13px;
      display:flex; gap:8px; align-items:center;
    }

    /* responsive & small flourishes */
    @media (max-width:520px){
      .card{ padding:20px; border-radius:14px; }
      .badge{ width:52px; height:52px; font-size:18px; border-radius:12px; }
      .message{ font-size:15px; padding:18px; border-radius:12px; }
      .heart{ width:48px; height:48px; font-size:20px; border-radius:10px; }
    }

    /* gentle entrance animation */
    .card{ transform:translateY(8px) scale(.995); opacity:0; animation:pop .6s cubic-bezier(.2,.9,.25,1) forwards; }
    @keyframes pop{
      to{ transform:none; opacity:1; }
    }

    /* small sparkle */
    .sparkle{
      display:inline-block;
      color:#c28bff;
      transform-origin:center;
      animation:twinkle 2.2s infinite;
    }
    @keyframes twinkle{
      0%{ transform:translateY(0) scale(.98); opacity:.9; }
      50%{ transform:translateY(-3px) scale(1.06); opacity:1; }
      100%{ transform:translateY(0) scale(.98); opacity:.9; }
    }

    /* subtle drop-in for the ask text */
    .ask{
      margin-top:14px;
      font-weight:600;
      color:#0b2b3a;
      font-size:16px;
    }

    .small-note{
      margin-top:8px;
      font-size:13px;
      color:var(--muted);
    }

  </style>
</head>
<body>
  <div class="wrap">
    <article class="card" role="article" aria-labelledby="title">
      <section class="message" aria-label="message">
        <span class="quote">Mitch, My Love,</span>

        <div class="soft">
          this is just a simple gift but it carries a piece of how I see you. I know this might not perfectly match every detail of your aesthetic, pero habang pinipili ko siya, ang nasa isip ko lang ay yung softness mo, yung strength mo, at yung paraan mo na nagiging “home” sa gulo ng araw ko.
        </div>

        <div style="height:10px"></div>

        <div class="soft">
          I remember you mentioned rings with a little sparkle on the edge .. So I chose something that reminds me of the way you shine quietly ... hindi maingay, pero napaka noticeable.
        </div>

        <div style="height:10px"></div>

        <div class="soft">
          This isn’t just a thing. It’s a small promise .. a reminder that I’m learning you piece by piece .. your moods, your storms, your quiet victories. and I’m choosing you in every version.
        </div>

        <div style="height:12px"></div>

        <div class="ask">
          And if you’re okay with it, I wanna ask gently:
        </div>

        <div style="height:6px"></div>

        <div style="font-size:18px; font-family: 'Playfair Display', serif; color:#081325;">
          <strong>Pwede na ba kitang tawagin na akin .. officially?</strong><br/><br/>
          <strong>Pwede na ba kitang maging girlfriend, Love?</strong>
          <span class="sparkle">✨</span>
        </div>

        <div class="small-note">
          If your heart says yes, I’ll treasure it gently. If it needs time, I’ll wait with calm and gratitude. Whatever your answer .. thank you for letting me care for you this far. Thank you for being you. 😘
        </div>

      </section>
    </article>
  </div>
</body>
</html>
