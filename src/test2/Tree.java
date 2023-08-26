package test2;

import java.util.*;
import java.util.stream.Collectors;

class Node {
    public int val;
    public Node left;
    public Node right;
    public Node next;

    public Node() {
    }

    public Node(int _val) {
        val = _val;
    }

    public Node(int _val, Node _left, Node _right, Node _next) {
        val = _val;
        left = _left;
        right = _right;
        next = _next;
    }
};

class TreeNode {
    int val;
    TreeNode left;
    TreeNode right;

    TreeNode() {
    }

    TreeNode(int val) {
        this.val = val;
    }

    TreeNode(int val, TreeNode left, TreeNode right) {
        this.val = val;
        this.left = left;
        this.right = right;
    }
}

public class Tree {
    public TreeNode buildBylevel(int[] nums) {
        TreeNode root = new TreeNode(nums[0]);
        Deque<TreeNode> q = new LinkedList<>();
        q.offer(root);
        int i = 1, n = nums.length;
        while (!q.isEmpty()) {
            TreeNode t = q.poll();
            if (i < n && nums[i] != -1 && q.offer(t.left = new TreeNode(nums[i]))) ;
            i++;
            if (i < n && nums[i] != -1 && q.offer(t.right = new TreeNode(nums[i]))) ;
            i++;
        }
        return root;
    }

    //    //     94. 二叉树的中序遍历
    public List<Integer> inorderTraversal(TreeNode root) {
        return inorderTraversal2(root);
    }

    //    //    非递归
    public List<Integer> inorderTraversal2(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        Stack<TreeNode> stack = new Stack<>();
//        stack.push(root);
        while (!stack.isEmpty() || root != null) {
            while (root != null) {
                stack.push(root);
                root = root.left;
            }
            res.add(stack.peek().val);
            root = stack.pop().right;
        }
        return res;
    }

    //
//    ///后序非递归
//    public List<Integer> postOrder(TreeNode root) {
//        List<Integer> res = new ArrayList<>();
//        Stack<TreeNode> stack = new Stack<>();
//        stack.push(root);
//        TreeNode pre = null, now;
//        while (!stack.isEmpty()) {
//            now = stack.peek();
//            if (Objects.isNull(now.left) && Objects.isNull(now.right) || (Objects.nonNull(pre) && (Objects.equals(now.left, pre) || Objects.equals(now.right, pre)))) {
//                res.add(now.val);
//                pre = stack.pop();
//            } else {
//                if (Objects.nonNull(now.right)) stack.push(now.right);
//                if (Objects.nonNull(now.left)) stack.push(now.left);
//            }
//        }
//        return res;
//    }

    //    //后序非递归（栈存放祖先节点）
    public List<Integer> postOrder2(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        Stack<TreeNode> stack = new Stack<>();
        TreeNode pre = null, now;
        while (Objects.nonNull(root) || !stack.isEmpty()) {
            if (Objects.nonNull(root)) {
                stack.push(root);
                root = root.left;
            } else {
                now = stack.peek();
                //Objects.nonNull(now.right)在访问左节点的时候会用到，now.right右边为空说明now没有哦右边也就是now该出来了  不能进入这个if  因为左右根的顺序 右边没有
                // !Objects.equals(now.right, pre)第二个条件是在now这个点二次被访问的时候，也就是now的右节点刚被访问完 此时now也该出来了 不能进入这个if 因为左右根的顺序
                if (Objects.nonNull(now.right) && !Objects.equals(now.right, pre)) {
                    root = now.right;
                } else {
                    res.add(stack.pop().val);
                    pre = now;
//                    root = null;
                }

            }
        }
        return res;
    }

    //
//    //101. 对称二叉树
    public boolean isSymmetric(TreeNode root) {
        Deque<TreeNode> q = new LinkedList<TreeNode>();
        q.push(root);
        q.push(root);
        TreeNode l, r;
        while (!q.isEmpty()) {
            r = q.pop();
            l = q.pop();
            if (l == null && r == null) {
                continue;
            }
            if (l == null || r == null || l.val != r.val) {
                return false;
            }
            q.offer(l.left);
            q.offer(r.right);
            q.offer(l.right);
            q.offer(r.left);

        }
        return true;
    }

    public void inorderTraversalDG(TreeNode root, List<Integer> res) {
        if (Objects.nonNull(root)) {
            inorderTraversalDG(root.left, res);
            res.add(root.val);
            inorderTraversalDG(root.right, res);
        }
    }

    //
//    //104. 二叉树的最大深度
    public int maxDepth(TreeNode root) {
        if (Objects.isNull(root)) {
            return 0;
        }
        int max = 0;
        Stack<TreeNode> stack = new Stack<TreeNode>();
        TreeNode cur, pre = null;
        while (!stack.isEmpty() || root != null) {
            while (root != null) {
                stack.push(root);
                max = Math.max(max, stack.size());
                root = root.left;
            }
            cur = stack.peek();
            if (cur.right != null && cur.right != pre) {
                root = cur.right;
            } else {
                pre = stack.pop();
            }
        }

        return max;
    }

    //    //108. 将有序数组转换为二叉搜索树
    public TreeNode sortedArrayToBST(int[] nums) {
        return helper(nums, 0, nums.length - 1);

    }

    public TreeNode helper(int[] nums, int left, int right) {
        if (left > right) return null;
        int mid = (left + right) / 2;
        TreeNode root = new TreeNode(nums[mid]);
        root.left = helper(nums, left, mid - 1);
        root.right = helper(nums, mid + 1, right);
        return root;
    }

    //    //98. 验证二叉搜索树
    public boolean isValidBST(TreeNode root) {
        return isValidBST(root, Long.MIN_VALUE, Long.MAX_VALUE);
    }

    //    非递归版本 中序非递归解决
    public boolean isValidBST2(TreeNode root) {
        Stack<TreeNode> stack = new Stack<>();
        TreeNode pre = null;
        while (!stack.isEmpty() || root != null) {
            while (root != null) {
                stack.push(root);
                root = root.left;
            }
            if (pre != null && pre.val >= stack.peek().val) {
                return false;
            }
            pre = stack.peek();
            root = stack.pop().right;
        }
        return true;
    }

    public boolean isValidBST(TreeNode root, long lower, long upper) {
        if (Objects.isNull(root)) {
            return true;
        }
        if (root.val <= lower || root.val >= upper) {
            return false;
        }
        return isValidBST(root.left, lower, root.val) && isValidBST(root.right, root.val, upper);
    }

    //    //102. 二叉树的层序遍历
    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        if (Objects.isNull(root)) return res;
        Deque<TreeNode> q = new ArrayDeque<>();
        q.offer(root);
        while (!q.isEmpty()) {
            res.add(q.stream().map(e -> e.val).collect(Collectors.toList()));
            int len = q.size();
            while (len > 0) {
                TreeNode cur = q.pop();
                if (cur.left != null) {
                    q.offerLast(cur.left);
                }
                if (cur.right != null) {
                    q.offerLast(cur.right);
                }
                len--;
            }
        }

        return res;

    }

    //    //230. 二叉搜索树中第K小的元素
    public int kthSmallest(TreeNode root, int k) {
        Stack<TreeNode> stack = new Stack<TreeNode>();
        while (!stack.isEmpty() || root != null) {
            while (root != null) {
                stack.push(root);
                root = root.left;
            }
            if ((k = k - 1) == 0) {
                return stack.peek().val;
            }
            root = stack.pop().right;
        }
        return 0;

    }

    //
//    public Node connect(Node root) {
//        List<List<Integer>> res = new ArrayList<>();
//        if (Objects.isNull(root)) return root;
//        Queue<Node> q = new LinkedList<>();
//        q.offer(root);
//        while (!q.isEmpty()) {
//            List<Integer> tempres = new ArrayList<>();
//            int sz = q.size();
//            Node pre = q.peek();
//            while (sz > 0) {
//                Node now = q.poll();
////                tempres.add(now.val);
//                if (Objects.nonNull(now.left)) q.offer(now.left);
//                if (Objects.nonNull(now.right)) q.offer(now.right);
//                pre.next = now;
//                pre = now;
//                sz--;
//                if (sz == 0) now.next = null;
//            }
//            res.add(tempres);
//        }
//        return root;
//
//    }
//
//    //236. 二叉树的最近公共祖先
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
//        Comparator
        Stack<TreeNode> stack = new Stack<TreeNode>();
        Set<TreeNode> set = new HashSet<>();
        TreeNode cur, pre = null, r = root;
        while (!stack.isEmpty() || r != null) {
            while (r != null) {
                stack.push(r);
                if (r == p || r == q) {
                    if (set.isEmpty()) {
                        set.addAll(new HashSet<>(stack));
                    } else {
                        while (!stack.isEmpty()) {
                            if (set.contains(stack.peek())) {
                                return stack.pop();
                            }
                            stack.pop();
                        }
                    }
                }
                r = r.left;
            }
            cur = stack.peek();
            if (cur.right != null && cur.right != pre) {
                r = cur.right;
            } else {
                pre = stack.pop();
            }
        }
        return root;

    }

    //
//
//    //226. 翻转二叉树
    public TreeNode invertTree(TreeNode root) {
        Stack<TreeNode> stack = new Stack<>();
        TreeNode p = root, top, pre = null, cur;
        while (Objects.nonNull(p) || !stack.isEmpty()) {
            while (Objects.nonNull(p)) {
                stack.push(p);
                p = p.left;
            }
            cur = stack.peek();
            if (Objects.nonNull(cur.right) && !Objects.equals(cur.right, pre)) {
                p = cur.right;
            } else {
                pre = stack.pop();
                TreeNode t = pre.left;
                pre.left = pre.right;
                pre.right = t;
            }
        }
        return root;

    }

    //
    public TreeNode invertTree2(TreeNode root) {
        Stack<TreeNode> stack = new Stack<>();
        TreeNode p = root, top, pre = null;
        while (Objects.nonNull(p) || !stack.isEmpty()) {
            if (Objects.nonNull(p)) {
                stack.push(p);
                p = p.left;
            } else {
                top = stack.pop();
                p = top.right;
                TreeNode t = top.left;
                top.left = top.right;
                top.right = t;

            }
        }
        return root;

    }

    ////    剑指 Offer 55 - I. 二叉树的深度
    public int maxDepth2(TreeNode root) {
        Stack<TreeNode> stack = new Stack<TreeNode>();
        TreeNode cur, pre = null;
        int max = 0;
        while (!stack.isEmpty() || root != null) {
            while (root != null) {
                stack.push(root);
                max = Math.max(max, stack.size());
                root = root.left;
            }
            cur = stack.peek();
            if (cur.right != null && cur.right != pre) {
                root = cur.right;
            } else {
                pre = stack.pop();
            }
        }
        return max;

    }

    //
    int max = 0;

    //
    public int diameterOfBinaryTree(TreeNode root) {
        preOrder(root);
        return max - 1;
    }

    public int preOrder(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int lm = preOrder(root.left);
        int rm = preOrder(root.right);
        max = Math.max(max, lm + rm + 1);
        return Math.max(lm, rm) + 1;
    }

    //    //617. 合并二叉树
    public TreeNode mergeTrees(TreeNode root1, TreeNode root2) {
        if (Objects.isNull(root1) && Objects.isNull(root2)) {
            return null;
        }
        if (Objects.isNull(root2)) {
            return root1;
        }
        if (Objects.isNull(root1)) {
            root1 = new TreeNode();
        }
        root1.val += root2.val;
        root1.left = mergeTrees(root1.left, root2.left);
        root1.right = mergeTrees(root1.right, root2.right);
        return root1;
    }

    public TreeNode mergeTrees(TreeNode root1, TreeNode root2, TreeNode root3, int n) {
        if (Objects.nonNull(root1) || Objects.nonNull(root2)) {
//            if (Objects.nonNull(root1))
            Integer val = 0;
            if (Objects.nonNull(root2)) {
                val += root2.val;
            }
            if (Objects.nonNull(root1)) {
                val += root1.val;
            }
//            MyUtile.printBlank(n++,val.toString());

            root3 = new TreeNode(val);
            root3.left = mergeTrees(Optional.ofNullable(root1).map(r -> r.left).orElse(null), Optional.ofNullable(root2).map(r -> r.left).orElse(null), null, n);
            root3.right = mergeTrees(Optional.ofNullable(root1).map(r -> r.right).orElse(null), Optional.ofNullable(root2).map(r -> r.right).orElse(null), null, n);
//            MyUtile.printBlank(--n,val.toString());
            return root3;
        } else {
            return null;
        }
    }

    //
//    //    111. 二叉树的最小深度 bfs
    public int minDepth(TreeNode root) {
        if (root == null) {
            return 0;
        }
        Stack<TreeNode> s = new Stack<TreeNode>();
        TreeNode cur, pre = null;
        int min = 9999;
        while (!s.isEmpty() || root != null) {
            while (root != null) {
                s.push(root);
                root = root.left;
            }
            cur = s.peek();
            if (cur.left == null && cur.right == null) {
                min = Math.min(min, s.size());
            }
            if (cur.right != null && cur.right != pre) {
                root = cur.right;
            } else {
                pre = s.pop();
            }
        }
        return min;
    }

    //
//    //    752. 打开转盘锁 bfs
    public int openLock(String[] deadends, String target) {
        List<String> dead = Arrays.stream(deadends).collect(Collectors.toList());
        Deque<String> q = new LinkedList<>();
        Set<String> v = new HashSet<>();
        q.offer("0000");
        v.add("0000");
        int len, res = 0;
//        DelayQueue
        while (!q.isEmpty()) {
            len = q.size();
            while ((len = len - 1) > 0) {
                String cur = q.pop();
//                if (Objects.equals(cur,d))
                if (dead.contains(cur)) {
                    continue;
                }
                if (Objects.equals(cur, target)) {
                    return res;
                }
                for (int i = 0; i < 4; i++) {
                    String t = add1(cur, i);
                    if (!v.contains(t)) {
                        q.offer(t);
                        v.add(t);
                    }
                    t = less1(cur, i);
                    if (!v.contains(t)) {
                        q.offer(t);
                        v.add(t);
                    }
                }
            }
            res++;
        }
        return -1;
    }

    //
    public String add1(String s, int i) {
        StringBuilder ss = new StringBuilder(s);
        if (ss.charAt(i) == '9') {
            ss.setCharAt(i, '0');
        } else {
            ss.setCharAt(i, (char) (ss.charAt(i) + 1));
        }
        return ss.toString();
    }

    public String less1(String s, int i) {
        StringBuilder ss = new StringBuilder(s);
        if (ss.charAt(i) == '0') {
            ss.setCharAt(i, '9');
        } else {
            ss.setCharAt(i, (char) (ss.charAt(i) - 1));
        }
        return ss.toString();
    }
//337. 打家劫舍 III

    public int rob(TreeNode root) {
        return rob(root, new HashMap<>());
    }

    public int rob(TreeNode root, Map<TreeNode, Integer> memo) {
        if (root == null) return 0;
        if (memo.containsKey(root)) {
            return memo.get(root);
        }
        if (root.left == null && root.right == null) return root.val;
        int l = rob(root.left, memo);
        int r = rob(root.right, memo);
        int lc = root.left == null ? 0 : rob(root.left.left, memo) + rob(root.left.right, memo);
        int rc = root.right == null ? 0 : rob(root.right.left, memo) + rob(root.right.right, memo);
        memo.put(root, Math.max(root.val + lc + rc, l + r));
        return memo.get(root);
    }

    //
    public static void main(String[] args) {
        Tree o = new Tree();
        TreeNode root = o.buildBylevel(new int[]{4, 2, -1, 1, 3});
        TreeNode root2 = o.buildBylevel(new int[]{2, 1, 3, -1, 4, -1, 7});

//        System.out.println(o.inorderTraversal2(root1));
//        System.out.println(o.maxDepth(root));
//        System.out.println(o.isValidBST(root));
//        System.out.println(o.levelOrder(root));
//        System.out.println(o.lowestCommonAncestor(root, new TreeNode(5), new TreeNode(4)));
//        System.out.println(o.postOrder(root));
//        System.out.println(o.diameterOfBinaryTree(root));
//        o.mergeTrees(root1, root2);
//        System.out.println(o.openLock(new String[]{"8888"}, "0009"));
//        System.out.println(o.openLock(new String[]{"8887","8889","8878","8898","8788","8988","7888","9888"}, "8888"));
        System.out.println(o.rob(root));
    }

}
