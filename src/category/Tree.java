package category;

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
    //112. 路径总和
    public boolean hasPathSum(TreeNode root, int targetSum) {
        if (root == null) {
            return false;
        }

        if (root.left == null && root.right == null) {
            return targetSum - root.val == 0;
        }
        return hasPathSum(root.left, targetSum - root.val) || hasPathSum(root.right, targetSum - root.val);
    }


//    2023.9.3之前:


    public TreeNode buildBylevel(TreeNode root, int[] nums) {
        if (nums.length == 0) {
            return null;
        }
        root = new TreeNode(nums[0]);
        TreeNode r = root;
        Queue<TreeNode> queue = new LinkedList<TreeNode>();
        queue.offer(r);
        int i = 1;
        while (!queue.isEmpty()) {
            TreeNode tmp = queue.poll();
            if (i < nums.length && nums[i] != -1) {
                tmp.left = new TreeNode(nums[i]);
                queue.offer(tmp.left);
            }
            i++;
            if (i < nums.length && nums[i] != -1) {
                tmp.right = new TreeNode(nums[i]);
                queue.offer(tmp.right);
            }
            i++;
        }
        return root;
    }


    //     94. 二叉树的中序遍历
    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        inorderTraversalDG(root, res);
        return res;
    }

    //    非递归
    public List<Integer> inorderTraversal2(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        Stack<TreeNode> stack = new Stack<>();
        while (Objects.nonNull(root) || !stack.isEmpty()) {
            while (Objects.nonNull(root)) {
                stack.push(root);
                root = root.left;
            }
            res.add(stack.pop().val);
            root = root.right;
        }
        return res;
    }

    ///后序非递归
    public List<Integer> postOrder(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        Stack<TreeNode> stack = new Stack<>();
        stack.push(root);
        TreeNode pre = null, now;
        while (!stack.isEmpty()) {
            now = stack.peek();
            if (Objects.isNull(now.left) && Objects.isNull(now.right) || (Objects.nonNull(pre) && (Objects.equals(now.left, pre) || Objects.equals(now.right, pre)))) {
                res.add(now.val);
                pre = stack.pop();
            } else {
                if (Objects.nonNull(now.right)) stack.push(now.right);
                if (Objects.nonNull(now.left)) stack.push(now.left);
            }
        }
        return res;
    }

    //后序非递归（栈存放祖先节点）
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

    //    145. 二叉树的后序遍历
    public List<Integer> postorderTraversal(TreeNode root) {
        List<TreeNode> res = new ArrayList<>();
        Stack<TreeNode> stack = new Stack<>();
        TreeNode pre = null;
        while (!stack.isEmpty() || root != null) {
            while (root != null) {
                stack.push(root);
                root = root.left;
            }
            TreeNode cur = stack.peek();
            if (cur.right != null && cur.right != pre) {
                root = cur.right;
            } else {
                res.add(pre = stack.pop());
            }
        }
        return res.stream().map(e -> e.val).collect(Collectors.toList());

    }

    //101. 对称二叉树
    public boolean isSymmetric(TreeNode root) {
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        queue.offer(root);
        while (!queue.isEmpty()) {
            TreeNode u = queue.poll();
            TreeNode v = queue.poll();
            if (Objects.isNull(u) && Objects.isNull(v)) {
                continue;
            }
            if ((Objects.isNull(u) || Objects.isNull(v)) || !Objects.equals(u.val, v.val)) {
                return false;
            }
            queue.offer(u.left);
            queue.offer(v.right);
            queue.offer(u.right);
            queue.offer(v.left);
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

    //104. 二叉树的最大深度
    public int maxDepth(TreeNode root) {

        int max = 0;

        if (Objects.isNull(root)) {
            return 0;
        }
        Queue<TreeNode> queue = new LinkedList<TreeNode>();
        queue.add(root);
        while (!queue.isEmpty()) {
            int sz = queue.size();
            while (sz > 0) {
                root = queue.poll();
                if (Objects.nonNull(root.left)) {
                    queue.offer(root.left);
                }
                if (Objects.nonNull(root.right)) {
                    queue.offer(root.right);
                }
                sz--;
            }
            max++;
        }
        return max;
    }

    //108. 将有序数组转换为二叉搜索树
    public TreeNode sortedArrayToBST(int[] nums) {
        return helper(nums, 0, nums.length - 1);

    }

    public TreeNode helper(int[] nums, int left, int right) {
        if (left > right) {
            return null;
        }
        int mid = (left + right) / 2;
        TreeNode root = new TreeNode(nums[mid]);
        root.left = helper(nums, left, mid - 1);
        root.right = helper(nums, mid + 1, right);
        return root;
    }

    //98. 验证二叉搜索树
    public boolean isValidBST(TreeNode root) {

        if (Objects.isNull(root)) {
            return true;
        }
        Stack<TreeNode> stack = new Stack<TreeNode>();

//        stack.push(root);
        long pre = -2147483649L;
        while (Objects.nonNull(root) || !stack.isEmpty()) {
            while (Objects.nonNull(root)) {
////                if (Objects.isNull(root.left) && Objects.isNull(root.right)) {
////                    continue;
////                }
//                if (Objects.isNull(root.left) && Objects.nonNull(root.right) && root.right.val <= root.val) {
//                    return false;
//                }
//                if (Objects.isNull(root.right) && Objects.nonNull(root.left) && root.left.val >= root.val) {
//                    return false;
//                }
//                if (Objects.nonNull(root.left) && Objects.nonNull(root.right) && (root.right.val <= root.val || root.left.val >= root.val || root.val)) {
//                    return false;
//                }
                stack.push(root);
                root = root.left;
            }
            root = stack.pop();
            if (root.val <= pre) {
                return false;
            }
            pre = root.val;
            ;
            root = root.right;
        }
        return true;
    }

    //102. 二叉树的层序遍历
    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        if (Objects.isNull(root)) return res;
        Queue<TreeNode> q = new LinkedList<>();
        q.offer(root);
        while (!q.isEmpty()) {
            List<Integer> tempres = new ArrayList<>();
            int sz = q.size();
            while (sz > 0) {
                TreeNode tmp = q.poll();
                tempres.add(tmp.val);
                if (Objects.nonNull(tmp.left)) q.offer(tmp.left);
                if (Objects.nonNull(tmp.right)) q.offer(tmp.right);
                sz--;
            }
            res.add(tempres);

        }
        return res;

    }

    //103. 二叉树的锯齿形层序遍历
    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {

        Deque<TreeNode> q = new LinkedList<>();
        List<List<Integer>> res = new ArrayList<>();
        if (root == null) {
            return res;
        }
        boolean left = true;
        q.offer(root);
        while (!q.isEmpty()) {
            int len = q.size() + 1;
            List<Integer> t = new ArrayList<TreeNode>(q).stream().map(e -> e.val).collect(Collectors.toList());
            if (!left) {
                Collections.reverse(t);
            }
            res.add(t);
            left = !left;

            while ((len = len - 1) > 0) {
                TreeNode node = q.poll();
                if (node.left != null) q.offer(node.left);
                if (node.right != null) q.offer(node.right);
            }
        }
        return res;
    }

    //    剑指 Offer 26. 树的子结构
    public boolean isSubStructure(TreeNode A, TreeNode B) {
        if (B == null && A != null || A == null && B != null) {
            return false;
        }
        Stack<TreeNode> stack = new Stack<>();
        TreeNode cur, pre = null;
        boolean res = false;
        while (!stack.isEmpty() || A != null) {
            while (A != null) {
                if (A.val == B.val && allMatch(A, B)) {
                    return true;
                }
                stack.push(A);
                A = A.left;
            }
            cur = stack.peek();
            if (cur.right != null && cur.right != pre) {
                A = cur.right;
            } else {
                pre = stack.pop();
            }
        }
        return false;
    }

    public boolean allMatch(TreeNode A, TreeNode B) {
        Stack<TreeNode> stack = new Stack<TreeNode>();
        Stack<TreeNode> stack1 = new Stack<TreeNode>();
        TreeNode cur, cur1, pre = null, pre1 = null;

        while (!stack.isEmpty() || B != null) {
            while (B != null) {
                if (A == null || B.val != A.val) {
                    return false;
                }
                stack.push(B);
                stack1.push(A);
                B = B.left;
                A = A.left;
            }

            cur = stack.peek();
            cur1 = stack1.peek();
            if (cur.right != null && cur.right != pre) {
                B = cur.right;
                A = cur1.right;
            } else {
                pre = stack.pop();
                pre1 = stack1.pop();
            }
        }
        return true;
    }

    //    剑指 Offer 26. 树的子结构(递归版本)
    public boolean isSubStructure1(TreeNode A, TreeNode B) {
        if (B == null) {
            return false;
        }
        if (A == null) {
            return false;
        }
        if (A.val == B.val && allMatch1(A, B)) {
            return true;
        }
        return isSubStructure1(A.left, B) || isSubStructure1(A.right, B);

    }

    public boolean allMatch1(TreeNode A, TreeNode B) {
        if (B == null) {
            return true;
        }
        if (A == null && B != null || A.val != B.val) {
            return false;
        }
        return allMatch1(A.left, B.left) && allMatch1(A.right, B.right);
    }


    //    199. 二叉树的右视图
    public List<Integer> rightSideView(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        rightSidePreOrder(root, res, 0, new HashSet<>());
        return res;
    }

    public void rightSidePreOrder(TreeNode root, List<Integer> res, int level, Set<Integer> v) {
        if (root == null) {
            return;
        }
        if (v.add(level) && res.add(root.val)) ;
        rightSidePreOrder(root.right, res, level + 1, v);
        rightSidePreOrder(root.left, res, level + 1, v);
    }

    //    199. 二叉树的右视图(非递归)
    public List<Integer> rightSideView1(TreeNode root) {
        Stack<TreeNode> nodeStack = new Stack<>();
        Stack<Integer> deepStack = new Stack<>();
        List<Integer> res = new ArrayList<>();
        Set<Integer> set = new HashSet<>();
        int level = 0;
        while (!nodeStack.isEmpty() || root != null) {
            while (root != null) {
                level++;
                nodeStack.push(root);
                deepStack.push(level);
                if (set.add(level) && res.add(root.val)) ;
                root = root.right;
            }
            root = nodeStack.pop().left;
            level = deepStack.pop();//在出栈的时候记录这个level，因为下一个点肯定是基于当前pop出来的的这个点的孩子节点，肯定是在这个level的基础上加一
        }
        return res;
    }

    //105. 从前序与中序遍历序列构造二叉树
    Map<Integer, Integer> map = new HashMap<>();

    public TreeNode buildTree(int[] preorder, int[] inorder) {
        for (int i = 0; i < preorder.length; i++) {
            map.put(inorder[i], i);
        }
        return buildTree(preorder, 0, preorder.length - 1, inorder, 0, inorder.length - 1);
    }

    public TreeNode buildTree(int[] preorder, int preL, int preR, int[] inorder, int inL, int inR) {
        if (preL > preR || inL > inR) {
            return null;
        }
        TreeNode root = new TreeNode(preorder[preL]);
        if (preL == preR) {
            return root;
        }

        int len = map.get(root.val) - inL;
        root.left = buildTree(preorder, preL + 1, preL + len, inorder, inL, map.get(root.val) - 1);
        root.right = buildTree(preorder, preL + len + 1, preR, inorder, map.get(root.val) + 1, inR);
        return root;
    }


    //106. 从中序与后序遍历序列构造二叉树
    Map<Integer, Integer> postmap = new HashMap<>();

    public TreeNode buildTree2(int[] inorder, int[] postorder) {
        for (int i = 0; i < inorder.length; i++) {
            postmap.put(inorder[i], i);
        }
        return buildTree3(inorder, 0, inorder.length - 1, postorder, 0, postorder.length - 1);
    }

    public TreeNode buildTree3(int[] inorder, int inL, int inR, int[] postorder, int postL, int postR) {
        if (inL > inR || postL > postR) {
            return null;
        }
        TreeNode root = new TreeNode(postorder[postR]);
        if (postL == postR) {
            return root;
        }
        int ind = postmap.get(root.val);
        int len = postmap.get(root.val) - inL;
        root.left = buildTree3(inorder, inL, ind - 1, postorder, postL, postL + len - 1);
        root.right = buildTree3(inorder, ind + 1, inR, postorder, postL + len, postR - 1);
        return root;
    }

    //    654. 最大二叉树
    public TreeNode constructMaximumBinaryTree(int[] nums) {
        return constructMaximumBinaryTree(nums, 0, nums.length - 1);
    }

    public TreeNode constructMaximumBinaryTree(int[] nums, int l, int r) {
        if (l > r) {
            return null;
        }
        int ind = getMaxInd(nums, l, r);
        TreeNode root = new TreeNode(nums[ind]);
        if (l == r) {
            return root;
        }
        root.left = constructMaximumBinaryTree(nums, l, ind - 1);
        root.right = constructMaximumBinaryTree(nums, ind + 1, r);
        return root;
    }

    public int getMaxInd(int[] nums, int l, int r) {
        int max = nums[l], res = l;
        for (int i = l; i <= r; i++) {
            if (nums[i] > max) {
                res = i;
                max = nums[i];
            }
        }
        return res;
    }

    //889. 根据前序和后序遍历构造二叉树
    Map<Integer, Integer> prepostmap = new HashMap<>();

    public TreeNode constructFromPrePost(int[] preorder, int[] postorder) {
        for (int i = 0; i < postorder.length; i++) {
            prepostmap.put(postorder[i], i);
        }
        return constructFromPrePost(preorder, 0, preorder.length - 1, postorder, 0, postorder.length - 1);
    }

    public TreeNode constructFromPrePost(int[] preorder, int preL, int preR, int[] postorder, int postL, int postR) {
        if (preL > preR) {
            return null;
        }
        TreeNode root = new TreeNode(preorder[preL]);
        if (preL == preR) {
            return root;
        }
        //寻找左子树的根的下标，按照这个根进行划分
        int ind = prepostmap.get(preorder[preL + 1]);
        root.left = constructFromPrePost(preorder, preL + 1, preL + 1 + ind - postL, postorder, postL, ind);
        root.right = constructFromPrePost(preorder, preL + ind - postL + 2, preR, postorder, ind + 1, postR - 1);
        return root;
    }


    //230. 二叉搜索树中第K小的元素
    public int kthSmallest(TreeNode root, int k) {

        if (Objects.isNull(root)) {
            return 0;
        }
        Stack<TreeNode> stack = new Stack<>();
//        stack.push(root);
        int count = 1;
        while (Objects.nonNull(root) || !stack.isEmpty()) {
            while (Objects.nonNull(root)) {
                stack.push(root);
                root = root.left;
            }
            root = stack.pop();
            if (k == count++) {
                return root.val;
            }
            root = root.right;
        }
        return 0;
    }

    public Node connect(Node root) {
        List<List<Integer>> res = new ArrayList<>();
        if (Objects.isNull(root)) return root;
        Queue<Node> q = new LinkedList<>();
        q.offer(root);
        while (!q.isEmpty()) {
            List<Integer> tempres = new ArrayList<>();
            int sz = q.size();
            Node pre = q.peek();
            while (sz > 0) {
                Node now = q.poll();
//                tempres.add(now.val);
                if (Objects.nonNull(now.left)) q.offer(now.left);
                if (Objects.nonNull(now.right)) q.offer(now.right);
                pre.next = now;
                pre = now;
                sz--;
                if (sz == 0) now.next = null;
            }
            res.add(tempres);
        }
        return root;

    }

    //236. 二叉树的最近公共祖先 递归做法
    public TreeNode lowestCommonAncestor2(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null) {
            return null;
        }
        if (root == p || root == q) {
            return root;
        }
        TreeNode l = lowestCommonAncestor(root.left, p, q);
        TreeNode r = lowestCommonAncestor(root.right, p, q);
        if (l != null && r != null) {
            return root;
        }
        if (l != null) {
            return l;
        }
        return r;
    }

    public TreeNode lca(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null) {
            return null;
        }
        if (root == p || root == q) {
            return root;
        }
        TreeNode l = lca(root.left, p, q);
        TreeNode r = lca(root.right, p, q);
        if (l != null && r != null) {
            return root;
        }
        if (l != null) {
            return l;
        }
        return r;
    }

    //236. 二叉树的最近公共祖先 栈后序做法
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
//        Comparator
        Stack<TreeNode> stack = new Stack<>();
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


    //226. 翻转二叉树
    public TreeNode invertTree(TreeNode root) {
        Stack<TreeNode> stack = new Stack<>();
        TreeNode p = root, top, pre = null;
        while (Objects.nonNull(p) || !stack.isEmpty()) {
            if (Objects.nonNull(p)) {
                stack.push(p);
                p = p.left;
            } else {
                top = stack.peek();
                if (Objects.nonNull(top.right) && !Objects.equals(pre, top.right)) {
                    p = top.right;
                } else {
                    stack.pop();
                    pre = top;
                    TreeNode t = top.left;
                    top.left = top.right;
                    top.right = t;
                }
            }
        }
        return root;

    }

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
//    剑指 Offer 55 - I. 二叉树的深度

    public int maxDepth2(TreeNode root) {
        Stack<TreeNode> stack = new Stack<TreeNode>();
        TreeNode r = root, pre = null, top;
        int max = 0;
        while (Objects.nonNull(r) || !stack.isEmpty()) {
            if (Objects.nonNull(r)) {
                stack.push(r);
                max = Math.max(max, stack.size());
                r = r.left;
            } else {
                top = stack.peek();
                if (Objects.nonNull(top.right) && !Objects.equals(top.right, pre)) {
                    r = top.right;
                } else {
                    pre = stack.pop();
                }
            }
        }
        return max;

    }

    //543. 二叉树的直径
    int max = 0;

    public int diameterOfBinaryTree(TreeNode root) {
        preOrder(root);
        return max - 1;
    }

    public int preOrder(TreeNode root) {
        if (Objects.equals(null, root)) {
            return 0;
        }
        int l = preOrder(root.left);
        int r = preOrder(root.right);
        max = Math.max(max, l + r + 1);
//        System.out.println(max);
        return Math.max(l, r) + 1;
    }

    //    124. 二叉树中的最大路径和
    int maxSum = -999999;

    public int maxPathSum(TreeNode root) {
        postOrder3(root);
        return maxSum;
    }

    //    方法的含义是root节点下的这棵树的最大值
    public int postOrder3(TreeNode root) {
        if (Objects.equals(null, root)) {
            return -999999;
        }
        int l = postOrder3(root.left);
        int r = postOrder3(root.right);
        PriorityQueue<Integer> heap = new PriorityQueue<>((o1, o2) -> o2 - o1);
//        使用大根堆对六种情况取最大：
//        当前节点不连接的情况（当前为负数连接的话会降低总体的值）：单独左子树l，单独右子树r     当前节点连接的情况分三种（因为左右可能为负数，所以左右单独考虑）：左+root，右+root，左右+root    直接当前，左右都不连接：root
        heap.offer(l);
        heap.offer(r);
        heap.offer(root.val);
        heap.offer(l + r + root.val);
        heap.offer(l + root.val);
        heap.offer(r + root.val);
        maxSum = Math.max(maxSum, heap.peek());
        heap.clear();
//        return返回的是当前节点root必须连接的情况，就是当前这棵树的最大值 如果root不连接会导致root这个节点断掉
        return Math.max(Math.max(l, r) + root.val, root.val);
    }


    //617. 合并二叉树
    public TreeNode mergeTrees(TreeNode root1, TreeNode root2) {
        TreeNode root3 = mergeTrees(root1, root2, null, 0);
//        System.out.println(inorderTraversal(mergeTrees(root1, root2, null,0)));
        return root3;
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

    //    111. 二叉树的最小深度 bfs
    public int minDepth(TreeNode root) {
        if (Objects.isNull(root)) return 0;
//        Set<TreeNode> set=new HashSet<>();
        Deque<TreeNode> q = new LinkedList<>();

        q.offer(root);
//        set
        int res = 0;
        while (!q.isEmpty()) {
            int sz = q.size();
            res++;
            for (int i = 0; i < sz; i++) {
                TreeNode cur = q.pop();
                if (Objects.isNull(cur.left) && Objects.isNull(cur.right)) {
                    return res;
                }
                if (Objects.nonNull(cur.left) && q.offer(cur.left)) ;
                if (Objects.nonNull(cur.right) && q.offer(cur.right)) ;
            }
        }
        return res;

    }

    //    752. 打开转盘锁 bfs
    public int openLock(String[] deadends, String target) {
        List<String> dead = Arrays.stream(deadends).collect(Collectors.toList());
        Set<String> vi = new HashSet<>();
        Deque<String> q = new LinkedList<>();
        q.offer("0000");
        vi.add("0000");

        int res = 0;
        while (!q.isEmpty()) {
            int sz = q.size();

//            res++;
            for (int i = 0; i < sz; i++) {
                String cur = q.poll();

                if (dead.contains(cur)) {
                    continue;
                }
                if (cur.equals(target)) {
                    System.out.println(target + " ? " + cur);
                    System.out.println(res);
                    return res;
                }
                for (int j = 0; j < 4; j++) {
                    String t = add1(cur, j);
                    if (!vi.contains(t)) {
                        q.offer(t);
                        vi.add(t);
                    }
                    t = less1(cur, j);
                    if (!vi.contains(t)) {
                        q.offer(t);
                        vi.add(t);
                    }
                }
            }
            res++;
        }
        return -1;

    }

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

    public static void main(String[] args) {
        Tree o = new Tree();
        TreeNode root = o.buildBylevel(new TreeNode(), new int[]{3, 2, 1, 6, 0, 5});
        TreeNode root2 = o.buildBylevel(new TreeNode(), new int[]{9, 3, 15, 20, 7});

        System.out.println(o.levelOrder(o.constructMaximumBinaryTree(new int[]{3, 2, 1, 6, 0, 5})));
//        System.out.println(o.rightSideView1(root));
//        System.out.println(o.zigzagLevelOrder(root));
//        System.out.println(o.maxPathSum(root));
//        System.out.println(o.maxDepth(root));
//        System.out.println(o.isValidBST(root));
//        System.out.println(o.levelOrder(root));
//        System.out.println(o.lowestCommonAncestor(root, new TreeNode(5), new TreeNode(1)).val);
//        System.out.println(o.postOrder(root));
//        System.out.println(o.diameterOfBinaryTree(root));
//        o.mergeTrees(root1, root2);
//        System.out.println(o.openLock(new String[]{"8888"}, "0009"));
//        System.out.println(o.openLock(new String[]{"8887","8889","8878","8898","8788","8988","7888","9888"}, "8888"));
    }

}
