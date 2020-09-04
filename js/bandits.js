/*
 * Interactive Javascript applets to explore bandit algorithms
 *
 * This code defines two applets to explore the multi-armed bandit problem, 
 * and Thompson Sampling in particular. They are written for this blog post:
 *
 *   https://gertjanvandenburg.com/blog/thompson_sampling/
 *
 * The code for the EvaluationApplet depends on the LineGraph.js code 
 * available here:
 *
 *   https://github.com/GjjvdBurg/LineGraph.js
 *
 * This code is released for demonstration purposes only, in case you'd like 
 * to learn how to make Javascript applets such as these for yourself.
 *
 * Copyright 2020, G.J.J. van den Burg
 *
 */

/* Above copyright notice holds unless otherwise noted */

var DOMAIN = 3;

class BanditsApplet {

	constructor(selector, sliderIds) {
		this.selector = selector;
		this.nBandits = 10;
		this.env = null;
		this.ts = null;
		this.images = null;
		this.last_action = null;
		this.last_reward = null;

		this.appletRunning = false; // for pause/continue

		this.sliderIds = sliderIds;
		this.sliders = {};
		this.init_sliders();
	}

	init_sliders() {
		for (let [key, id] of Object.entries(this.sliderIds)) {
			this.sliders[key] = document.getElementById(id);
			if (this.sliders[key].hasOwnProperty('noUiSlider'))
				this.sliders[key].noUiSlider.destroy();
		}
		noUiSlider.create(this.sliders['prior_m'], {
			start: [0],
			range: {'min': -2, 'max': 2},
			step: 0.1,
			tooltips: [true],
		});
		noUiSlider.create(this.sliders['prior_v'], {
			start: [1],
			range: {'min': 0,
				'20%': [0.1, 0.1],
				'30%': [1, 1],
				'60%': [10, 10],
				'max': 100},
			tooltips: [true],
		});
		noUiSlider.create(this.sliders['prior_a'], {
			start: [1],
			range: {'min': [0],
				'10%': [0.1, 0.1],
				'20%': [1, 1],
				'30%': [10, 10],
				'50%': [100, 100],
				'max': [1000]},
			tooltips: [true],
		});
		noUiSlider.create(this.sliders['prior_b'], {
			start: [1],
			range: {'min': [0],
				'10%': [0.1, 0.1],
				'20%': [1, 1],
				'30%': [10, 10],
				'50%': [100, 100],
				'max': [1000]},
			tooltips: [true],
		});
		var self = this;
		for (let [key, id] of Object.entries(this.sliderIds)) {
			this.sliders[key].noUiSlider.on('change', function() {
				var wasRunning = self.appletRunning;
				self.reset(true);
				if (wasRunning) self.toggle();
			});
		}
	}

	init(outerWidth, outerHeight) {
		this.outerWidth = outerWidth;
		this.outerHeight = outerHeight;

		var self = this;
		// register the buttons
		document.getElementById('ts-btn-toggle').addEventListener('click',
			function() { self.toggle(); });
		document.getElementById('ts-btn-reset').addEventListener('click',
			function() {
				var wasRunning = self.appletRunning;
				self.reset();
				if (wasRunning) self.toggle();
			});
		this.reset();
	}

	build_table() {
		var lrmargin = Math.ceil(0.05 * this.outerWidth);
		var margin = {top: 0, right: lrmargin, bottom: 0, left: lrmargin};
		this.cvHeight = 20;
		this.cvWidth = Math.floor(this.outerWidth - margin.right - margin.left);

		this.images = new Array(this.nBandits);

		var vizDiv = document.getElementById(this.selector);
		var oldTable = vizDiv.querySelector('#viz-table');
		if (oldTable) oldTable.remove();

		var table = document.createElement('TABLE');
		table.id = 'viz-table';

		var tr, td, th, thead, tbody, el;

		var addhead = function(lbl) {
			th = document.createElement('TH');
			if (lbl) {
				el = document.createElement('SPAN');
				el.innerHTML = lbl;
				th.appendChild(el);
			}
			thead.appendChild(th);
		}
		var addcell = function(lbl, cls, id) {
			td = document.createElement('TD');
			td.setAttribute('class', cls);
			el = document.createElement('SPAN');
			el.innerHTML = lbl;
			el.setAttribute('id', id);
			td.appendChild(el);
			tr.appendChild(td);
		}
		var meanStr = function(means, idx) {
			var mi = means.indexOf(Math.max(...means));
			var s = means[idx].toFixed(3);
			if (idx == mi)
				s = `<u>${s}</u>`;
			return s;
		}

		thead = document.createElement("THEAD");
		addhead();
		addhead('N');
		addhead('Estimate');
		addhead('True');

		table.appendChild(thead);

		tbody = document.createElement('TBODY');

		var mu, sigma2;
		for (let i=0; i<this.nBandits; i++) {
			mu = this.ts.get_rho(i);
			sigma2 = this.ts.get_sigma2_map(i);

			tr = document.createElement('TR');

			// first column contains canvas
			td = document.createElement('TD');
			this.images[i] = new SVGImage(this.cvWidth, this.cvHeight, margin);
			this.images[i].init(td);
			this.images[i].update(mu, sigma2);
			td.setAttribute('class', 'col-cvs');
			tr.appendChild(td);

			addcell('0', 'col-cnt', 'viz-count-' + i);
			addcell('0.0', 'col-est', 'viz-est-' + i);
			addcell(meanStr(this.env.means, i), 'col-mean', 'viz-mean' + i);

			tbody.appendChild(tr);
		}

		table.appendChild(tbody);
		vizDiv.appendChild(table);

	}

	get_param() {
		this.m = parseFloat(this.sliders['prior_m'].noUiSlider.get());
		this.v = parseFloat(this.sliders['prior_v'].noUiSlider.get());
		this.alpha = parseFloat(this.sliders['prior_a'].noUiSlider.get());
		this.beta = parseFloat(this.sliders['prior_b'].noUiSlider.get());
		// make sure these hyperparameters are always positive
		var bump = function(a) { return a < 0.001 ? 0.001 : a; };
		this.v = bump(this.v);
		this.alpha = bump(this.alpha);
		this.beta = bump(this.beta);
	}

	toggle() {
		if (this.appletRunning) {
			this.appletRunning = false;
			document.getElementById('ts-btn-toggle').className = 'btn-play';
		} else {
			this.appletRunning = true;
			document.getElementById('ts-btn-toggle').className = 'btn-pause';
			this.animate();
		}
	}

	reset(soft) {
		if (soft === undefined) soft = false;
		var self = this;

		var soft_reset = function() {
			// reset everything except the test bed
			self.get_param();
			self.t = 0;
			self.ts = new ThompsonSampling(self.nBandits, self.m, self.v,
				self.alpha, self.beta);
			self.build_table();
		}

		var hard_reset = function() {
			if (self.t == 0) // double reset also resets sliders
				self.init_sliders();
			self.get_param();
			self.t = 0;
			self.ts = new ThompsonSampling(self.nBandits, self.m, self.v,
				self.alpha, self.beta);
			self.env = new TestBed(self.nBandits, null, true);
			self.build_table();
		}

		var reset = soft ? soft_reset : hard_reset;

		// A delay is used because otherwise there's a chance that the 
		// last animation frame updates our clean table.
		if (this.appletRunning) {
			this.appletRunning = false;
			window.setTimeout(reset, 100);
		} else {
			reset();
		}

		document.getElementById('ts-btn-toggle').className = 'btn-play';
	}

	step() {
		this.t += 1;
		// perform a single step of the simulation
		var action = this.ts.act();
		var reward = this.env.step(action);
		this.ts.record(action, reward);
		this.last_action = action;
		this.last_reward = reward;
	}

	draw() {
		var i, mu, sigma2, seq, el;
		i = this.last_action;
		mu = this.ts.get_rho(i);
		sigma2 = this.ts.get_sigma2_map(i);

		this.images[i].update(mu, sigma2, this.last_reward);

		el = document.getElementById('viz-count-' + i);
		el.innerHTML = parseInt(el.innerHTML) + 1;

		el = document.getElementById('viz-est-' + i);
		el.innerHTML = mu.toFixed(3);
	}

	animate() {
		var frame, paint, parent = this;

		paint = function() {
			parent.step();
			parent.draw();

			frame = window.requestAnimationFrame(paint);
			if (!parent.appletRunning) {
				window.cancelAnimationFrame(frame);
			}

		}
		frame = requestAnimationFrame(paint);
	}
}

class EvaluationApplet {

	constructor(selector, inputIds) {
		this.selector = selector;
		this.inputIds = inputIds;
		this.nBandits = 10;
		this.graph = null;

		this.appletRunning = false;
	}

	init(outerWidth, outerHeight) {
		this.outerWidth = outerWidth;
		this.outerHeight = outerHeight;

		var self = this;
		// register the buttons
		document.getElementById('eval-btn-toggle').addEventListener('click',
			function() { self.toggle(); });
		document.getElementById('eval-btn-reset').addEventListener('click',
			function() { self.reset(); });

		this.reset();
	}

	init_graph() {
		this.graph = new LineGraph(this.selector, null);

		var rawData = {
			"meta": {
				"xlabel": "Time",
				"ylabel": "Regret",
				"title": "",
			},
			"data": {
				"X": [],
				"series": []
			}
		};
		for (const name of this.banditNames) {
			rawData['data']['series'].push({'name': name, 'values': []});
		}
		this.graph.rawData = rawData;
		this.graph.dataReady = true;
		this.graph.realPreProcess();
		this.graph.build(this.outerWidth, this.outerHeight);
	}

	get_param() {
		this.param = {};

		var self = this;

		var intFromId = function(x) {
			return parseInt(document.getElementById(
				self.inputIds[x]).value);
		}
		var floatsFromId = function(x) {
			return document.getElementById(self.inputIds[x]).value
				.split(',').map(parseFloat);
		}
		var onlyUnique = function(value, index, self) {
			return self.indexOf(value) === index; }
		var onlyPos = function(value, index, self) { return value > 0; }

		// extract the parameters from the input fields
		this.param.repeats = intFromId('repeats');
		this.param.epsilons = floatsFromId('epsilon').filter(onlyUnique);
		this.param.cs = floatsFromId('c').filter(onlyUnique).filter(onlyPos);
		this.param.ms = floatsFromId('m').filter(onlyUnique);
		this.param.nus = floatsFromId('nu').filter(onlyUnique).filter(onlyPos);
		this.param.alphas = floatsFromId('alpha').filter(onlyUnique).filter(onlyPos);
		this.param.betas = floatsFromId('beta').filter(onlyUnique).filter(onlyPos);

		// construct all bandit configurations (cross product)
		this.param.banditConfig = [];
		for (const eps of this.param.epsilons)
			this.param.banditConfig.push(["EpsilonGreedy", [this.nBandits, eps]]);

		for (const c of this.param.cs)
			this.param.banditConfig.push(["UpperConfidenceBound", [this.nBandits, c]]);

		for (const m of this.param.ms)
			for (const nu of this.param.nus)
				for (const alpha of this.param.alphas)
					for (const beta of this.param.betas)
						this.param.banditConfig.push(
							["ThompsonSampling", [this.nBandits,
									m, nu, alpha, beta]]);

		this.banditClasses = {EpsilonGreedy, UpperConfidenceBound, ThompsonSampling};
		this.banditNames = [];
		this.banditConfig = {};

		// map each bandit config to a unique name, using their 
		// label() method.
		var tmp, name;
		for (var config of this.param.banditConfig) {
			tmp = new this.banditClasses[config[0]](...config[1]);
			name = tmp.label();
			this.banditNames.push(name);
			this.banditConfig[name] = config;
		}
	}

	toggle() {
		if (this.appletRunning) {
			this.appletRunning = false;
			document.getElementById('eval-btn-toggle').className = 'btn-play';
		} else {
			this.appletRunning = true;
			document.getElementById('eval-btn-toggle').className = 'btn-pause';
			this.get_param();
			this.animate();
		}
	}

	reset() {
		var self = this;

		var real_reset = function() {
			self.t = 0;
			self.get_param();

			// initialize the regret value for each bandit (last 
			// iter), as well as the environment and bandit 
			// instance (replicated over the number of iterations)
			self.regret = {};
			self.envs = {};
			self.bandits = {};
			for (const name of self.banditNames) {
				self.regret[name] = 0.0;
				self.envs[name] = new Array(self.param.repeats);
				self.bandits[name] = new Array(self.param.repeats);
			}

			var seed, config;
			var alea = new Alea();

			for (let r=0; r<self.param.repeats; r++) {
				// ensure the different bandits operate on the 
				// same environment for the same repetition 
				// index.
				seed = alea.randint();
				for (const name of self.banditNames) {
					self.envs[name][r] = new TestBed(self.nBandits, seed);
					config = self.banditConfig[name];
					self.bandits[name][r] = new self.banditClasses[config[0]](...config[1]);
				}
			}

			self.init_graph();
		}

		// A delay is used because otherwise there's a chance that the 
		// last animation frame updates our graph
		if (this.appletRunning) {
			this.appletRunning = false;
			document.getElementById('eval-btn-toggle').className = 'btn-play';
			window.setTimeout(real_reset, 100);
		} else {
			real_reset();
		}
	}

	step() {
		this.t += 1;

		var action, reward, t_regret = {};

		// regret for this iteration
		for (const name of this.banditNames)
			t_regret[name] = 0.0;

		// perform a single step of all the environments/bandits
		for (let r=0; r<this.param.repeats; r++) {
			for (const name of this.banditNames) {
				action = this.bandits[name][r].act();
				reward = this.envs[name][r].step(action);
				this.bandits[name][r].record(action, reward);
				t_regret[name] += this.envs[name][r].optimal - this.envs[name][r].means[action];
			}
		}

		// record the average regret over all repetitions
		for (name of this.banditNames)
			this.regret[name] += t_regret[name] / this.param.repeats;

	}

	draw() {
		this.graph.appendObservation(this.t, this.regret);
		this.graph.updateXYMinMax();

		// rebuild the graph. This may not be the most efficient, but 
		// seems to work quite well in practice
		if (this.t % 2 == 0)
			this.graph.build(this.outerWidth, this.outerHeight);
	}

	animate() {
		var frame, paint, parent = this;

		paint = function() {
			parent.step();
			parent.draw();

			frame = window.requestAnimationFrame(paint);
			if (!parent.appletRunning) {
				window.cancelAnimationFrame(frame);
			}
		}

		frame = requestAnimationFrame(paint);
	}
}

class SVGImage {

	constructor(width, height, margin) {
		this.width = width;
		this.height = height;
		this.margin = margin;
		this.scale = d3.scaleLinear()
			.domain([-DOMAIN, DOMAIN])
			.range([this.margin.left, this.width - this.margin.right]);
		this.svg = null;
	}

	init(element) {
		var color = this.getColor();
		this.svg = d3.select(element)
			.append('svg')
			.attr('width', this.width)
			.attr('height', this.height)
			.append('g')
			.attr('transform', `translate(0, ${this.height})`)
			.call(rampHorizontal(this.scale, color));
	}

	getColor(mu, sigma2) {
		if (mu === undefined) mu = 0;
		if (sigma2 === undefined) sigma2 = 1;

		return d3.scaleSequentialQuantile(d3.interpolateRdBu)
			.domain(Float32Array.from({length: 10000}, d3.randomNormal(mu,
				Math.sqrt(sigma2))));
	}

	update(mu, sigma2, latest_reward) {
		var color = this.getColor(mu, sigma2);
		this.svg.call(rampHorizontal(this.scale, color));
	}
}

class CanvasImage {

	/* This was an alternative visualization for the normal distribution 
	 * in the bandits application, where we would add a pixel for each 
	 * draw of the posterior. It didn't quite work as nicely.
	 */
	constructor(width, height, margin) {
		this.width = width;
		this.height = height;
		this.margin = margin;

		this.canvas = null;
		this.ctx = null;
		this.data = null;
		this.rng = new RNG();
	}

	init(parent) {
		this.canvas = document.createElement('canvas');
		this.canvas.width = this.width;
		this.canvas.height = this.height;
		this.ctx = this.canvas.getContext('2d');
		this.data = this.ctx.getImageData(0, 0, this.width, this.height);
		parent.appendChild(this.canvas);
	}

	drawPixel(x, y, r, g, b, a) {
		var idx = (x + y * this.width) * 4;
		this.data.data[idx + 0] = r;
		this.data.data[idx + 1] = g;
		this.data.data[idx + 2] = b;
		this.data.data[idx + 3] += a;
	}

	update(mu, sigma, latest_reward) {
		if (latest_reward === undefined || Math.abs(latest_reward) > DOMAIN)
			return; // drop out of bounds
		var x = Math.floor((latest_reward + DOMAIN)/(2*DOMAIN) * this.width);
		for (let y=0; y<this.height; y++)
			this.drawPixel(x, y, 255, 0, 0, 17);
		this.ctx.putImageData(this.data, 0, 0);
	}
}

class TestBed {

	constructor(nBandits, seed, do_clamp) {
		this.nBandits = nBandits;
		this.rng = new RNG(seed);
		this.means = null;
		this.optimal = null;
		this.do_clamp = do_clamp === undefined ? false : do_clamp;

		this.stdevs = null;
		this.reset();

	}

	reset() {
		this.means = Array.from({length: this.nBandits},
			(v, i) => this.rng.gauss(0, 1));
		this.stdevs = Array.from({length: this.nBandits},
			(v, i) => this.rng.unif(0.5, 4.0));
		while (
			this.do_clamp &&
			Math.max.apply(null, this.means.map(Math.abs)) > DOMAIN
		) {
			this.means = Array.from({length: this.nBandits},
			(v, i) => this.rng.gauss(0, 1));
		}
		this.optimal = Math.max(...this.means);
	}

	step(action) {
		return this.rng.gauss(this.means[action], this.stdevs[action]);
		//return this.rng.gauss(this.means[action], 1.0);
	}
}

class EpsilonGreedy {

	constructor(nBandits, epsilon) {
		this.nBandits = nBandits;
		this.epsilon = epsilon;
		this.reset();
		this.rng = new RNG();
	}

	reset() {
		this.t = 0;
		this.Q = new Array(this.nBandits).fill(0.0);
		this.N = new Array(this.nBandits).fill(0);
	}

	act() {
		if (this.rng.random() <= this.epsilon) {
			return this.rng.randint(0, this.nBandits - 1);
		}
		var qprime = -Infinity;
		var aprime = null;
		for (let a=0; a<this.nBandits; a++) {
			if (this.Q[a] > qprime) {
				aprime = a;
				qprime = this.Q[a];
			}
		}
		return aprime;
	}

	record(action, reward) {
		var A = action;
		var R = reward;
		this.N[A] += 1;
		this.Q[A] += (1.0/this.N[A]) * (R - this.Q[A]);
	}

	label() {
		return `ε-Greedy (ε = ${this.epsilon})`;
	}
}

class UpperConfidenceBound {

	constructor(nBandits, c) {
		this.nBandits = nBandits;
		this.c = c;
		this.reset();
	}

	reset() {
		this.t = 0;
		this.Q = new Array(this.nBandits).fill(0.0);
		this.N = new Array(this.nBandits).fill(0);
	}

	act() {
		this.t += 1;
		var V_a;
		var aprime = null;
		var Vprime = -Infinity;
		for (let a=0; a<this.nBandits; a++) {
			if (this.N[a] == 0)
				// pull each handle at least once
				return a;

			// compute this action's value
			V_a = this.Q[a] + this.c * Math.sqrt(
				Math.log(this.t) / this.N[a]);

			// find the best action
			if (V_a > Vprime) {
				Vprime = V_a;
				aprime = a;
			}
		}
		return aprime;
	}

	record(action, reward) {
		var A = action;
		var R = reward;
		this.N[A] += 1;
		this.Q[A] += (1.0/this.N[A]) * (R - this.Q[A]);
	}

	label() {
		return `UCB (c = ${this.c})`;
	}
}

class ThompsonSampling {

	constructor(nBandits, prior_m, prior_v, prior_alpha, prior_beta) {
		this.nBandits = nBandits;
		this.m_a = prior_m;
		this.v_a = prior_v;
		this.alpha_a = prior_alpha;
		this.beta_a = prior_beta;
		this.rng = new RNG();
		this.reset();
	}

	reset() {
		this.N = new Array(this.nBandits).fill(0);
		this.mean = new Array(this.nBandits).fill(0.0);
		this.rho = new Array(this.nBandits).fill(this.m_a);
		this.ssd = new Array(this.nBandits).fill(0.0);
		this.beta_t_a = new Array(this.nBandits).fill(this.beta_a);

		// initialize prior variance at the mode of the IG divided by 
		// the pseudocount (reflects prior parameters)
		this.latest_zeta = new Array(this.nBandits).fill(
			(this.beta_a / (this.alpha_a + 1))/this.v_a);
	}

	_draw_ig(alpha, beta) {
		return 1.0 / this.rng.gamma(alpha, 1.0/beta);
	}

	_draw_normal(mu, sigma2) {
		return this.rng.gauss(mu, Math.sqrt(sigma2));
	}

	act() {
		var sigma2_a, mu_a;
		var aprime = null;
		var muprime = -Infinity;
		for (let a=0; a<this.nBandits; a++) {
			sigma2_a = this._draw_ig(
				0.5 * this.N[a] + this.alpha_a,
				this.beta_t_a[a]
			);
			this.latest_zeta[a] = sigma2_a / (this.N[a] + this.v_a);

			mu_a = this._draw_normal(
				this.rho[a],
				this.latest_zeta[a]
			);

			if (mu_a > muprime) {
				aprime = a;
				muprime = mu_a;
			}
		}
		return aprime;
	}

	record(action, reward) {
		var A = action;
		var R = reward;
		var prevN = this.N[A];
		var prevMean = this.mean[A];

		this.N[A] += 1;
		this.mean[A] += 1 / this.N[A] * (R - this.mean[A]);
		this.rho[A] = ((this.v_a * this.m_a + this.N[A] * this.mean[A])/
			(this.v_a + this.N[A]));
		this.ssd[A] += (
			R * R + prevN * prevMean * prevMean -
			this.N[A] * this.mean[A] * this.mean[A]
		);
		this.beta_t_a[A] = (
			this.beta_a
			+ 0.5 * this.ssd[A]
			+ (
				this.N[A] * this.v_a *
				(this.mean[A] - this.m_a) *
				(this.mean[A] - this.m_a) /
				(2 * (this.N[A] + this.v_a))
			)
		);
	}

	/* external */

	get_rho(idx) {
		return this.rho[idx];
	}

	get_sigma2_map(idx) {
		return (this.beta_t_a[idx] / (0.5 * this.N[idx] + this.alpha_a + 1))/(this.N[idx] + this.v_a)

	}

	label() {
		return `ThompsonSampling (m = ${this.m_a}, ν = ${this.v_a}, α = ${this.alpha_a}, β = ${this.beta_a})`;
	}
}

class RNG {

	constructor(seed) {
		// these are for the gamma distribution
		this.LOG4 = Math.log(4.0);
		this.SG_MAGICCONST = 1.0 + Math.log(4.5);

		// these are for the normal distribution
		this.TAU = 2 * Math.PI;
		this.z1 = null;
		this.generate = false;

		// use alea as underlying RNG so we can seed it
		this.seed = seed;
		this.alea = new Alea([seed]);
	}

	gauss(mu, sigma) {
		// sigma is standard deviation!
		this.generate = !this.generate;
		if (!this.generate)
			return this.z1 * sigma + mu;

		var u1 = 0, u2 = 0, z0;
		while (u1 === 0) u1 = this.alea.random();
		while (u2 === 0) u2 = this.alea.random();

		z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(this.TAU * u2);
		this.z1 = Math.sqrt(-2.0 * Math.log(u1)) * Math.sin(this.TAU * u2);
		return z0 * sigma + mu;
	}

	gamma(alpha, beta) {
		// adapted from CPython.

		if  (alpha <= 0.0 || beta <= 0.0)
			return null;

		if (alpha > 1.0) {
			var ainv = Math.sqrt(2.0 * alpha - 1.0);
			var bbb = alpha - this.LOG4;
			var ccc = alpha + ainv;

			var u1, u2, v, x, z, r;
			while (1) {
				u1 = this.alea.random();
				if (!(1e-7 < u1 && u1 < 0.9999999))
					continue

				u2 = 1.0 - this.alea.random();
				v = Math.log(u1 / (1.0 - u1))/ainv;
				x = alpha * Math.exp(v);
				z = u1 * u1 * u2;
				r = bbb + ccc * v - x;
				if (r + this.SG_MAGICCONST - 4.5*z >= 0.0 || r >= Math.log(z))
					return x * beta;
			}
		} else if (alpha == 1.0) {
			return -Math.log(1.0 - this.alea.random()) * beta;
		} else {
			var u, b, p, x, u1;
			while (1) {
				u = this.alea.random();
				b = (Math.E + alpha)/Math.E;
				p = b * u;
				if (p <= 1.0)
					x = Math.pow(p, 1.0/alpha);
				else
					x = -Math.log((b - p)/alpha);
				u1 = this.alea.random();
				if (p > 1.0) {
					if (u1 <= x ** (alpha - 1.0))
						break;
				} else if (u1 <= Math.exp(-x))
					break;
			}
			return x * beta;
		}
	}

	unif(a, b) {
		// return uniform on [a, b)
		return a + this.alea.random() * (b - a);
	}

	random() {
		return this.unif(0, 1);
	}

	randint(a, b) {
		// random integer on [a, b]
		var min = Math.ceil(a);
		var max = Math.floor(b);
		return Math.floor(this.alea.random() * (max - min + 1)) + min;
	}
}

// The following rampHorizontal function is adapted from here: 
// https://github.com/d3/d3-axis/issues/41
// The following license applies for this function.
/*
 * Copyright 2010-2016 Mike Bostock
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without 
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the 
 *    documentation and/or other materials provided with the distribution.
 *
 *  * Neither the name of the author nor the names of contributors may be used
 *    to endorse or promote products derived from this software without 
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE 
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
 * POSSIBILITY OF SUCH DAMAGE. 
 *
 */
rampHorizontal = function(x, color) {
	var size = 20; // should be the same as cvHeight for us

	function ramp(g) {
		var image = g.selectAll('image').data([null]),
			xz = x.range(),
			x0 = xz[0],
			x1 = xz[xz.length - 1],
			canvas = document.createElement('canvas'),
			context = (canvas.width = x1 - x0 + 1, canvas.height = 1, canvas).getContext('2d');

		for (let i=x0; i<=x1; ++i) {
			context.fillStyle = color(x.invert(i));
			context.fillRect(i - x0, 0, 1, 1);
		}

		image = image.enter().append('image').merge(image)
			.attr('x', x0)
			.attr('y', -size)
			.attr('width', x1 - x0 + 1)
			.attr('height', size)
			.attr('preserveAspectRatio', 'none')
			.attr('xlink:href', canvas.toDataURL());
	}

	ramp.position = function(_) {
		return arguments.length ? (x = _, ramp) : x;
	};

	ramp.color = function(_) {
		return arguments.length ? (color = _, ramp) : color ;
	};

	ramp.size = function(_) {
		return arguments.length ? (size = +_, ramp) : size;
	}

	return ramp;
};

// The following Alea class is a port of an algorithm by Johannes Baagøe 
// <baagoe@baagoe.com>, 2010 http://baagoe.com/en/RandomMusings/javascript/
// It is licensed under the following MIT license.
//
/*
 * Copyright (C) 2010 by Johannes Baagøe <baagoe@baagoe.org>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 */
class Alea {

	constructor(args) {
		this.s0 = 0;
		this.s1 = 0;
		this.s2 = 0;
		this.c = 1;

		if (args === undefined || args == null || args.length == 0 || args[0] === undefined || args[0] == null) {
			args = [+new Date];
		}

		var mash = this.Mash();
		this.s0 = mash(' ');
		this.s1 = mash(' ');
		this.s2 = mash(' ');

		for (let i=0; i<args.length; i++) {
			this.s0 -= mash(args[i]);
			if (this.s0 < 0) this.s0 += 1;
			this.s1 -= mash(args[i]);
			if (this.s1 < 0) this.s1 += 1;
			this.s2 -= mash(args[i]);
			if (this.s2 < 0) this.s2 += 1;
		}

		mash = null;
	}

	random() {
		var t = 2091639 * this.s0 + this.c * 2.3283064365386963e-10; // 2^-32
		this.s0 = this.s1;
		this.s1 = this.s2;
		return this.s2 = t - (this.c = t | 0);
	}

	randint() {
		return this.random() * 0x100000000; // 2^32
    	}

	Mash() {
		var n = 0xefc8249d;

		var mash = function(data) {
			data = String(data);
			for (let i=0; i<data.length; i++) {
				n += data.charCodeAt(i);
				var h = 0.02519603282416938 * n;
				n = h >>> 0;
				h -= n;
				h *= n;
				n = h >>> 0;
				h -= n;
				n += h * 0x100000000; // 2^32
			}
			return (n >>> 0) * 2.3283064365386963e-10; // 2^-3
		}

		return mash;
	}
}
